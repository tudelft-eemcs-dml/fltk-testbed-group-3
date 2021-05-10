import heapq
import time
from typing import List

import torch
from dataclass_csv import DataclassWriter
from pandas import np
from scipy import stats
from torch import nn
from torch.autograd import Variable
from torch.distributed import rpc

from fltk.client_fegan import Client
from fltk.strategy.client_selection import balanced_sampling, init_groups
from fltk.util.fed_avg import kl_weighting
from fltk.util.log import FLLogger
from fltk.nets.ls_gan import *
from fltk.util.weight_init import *
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging

from fltk.util.results import EpochData

logging.basicConfig(level=logging.DEBUG)


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


def _remote_method_async(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)


class ClientRef:
    ref = None
    name = ""
    data_size = 0
    tb_writer = None

    def __init__(self, name, ref, tensorboard_writer):
        self.name = name
        self.ref = ref
        self.tb_writer = tensorboard_writer

    def __repr__(self):
        return self.name


class Federator:
    """
    Central component of the Federated Learning System: The Federator

    The Federator is in charge of the following tasks:
    - Have a copy of the global model
    - Client selection
    - Aggregating the client model weights/gradients
    - Saving all the metrics
        - Use tensorboard to report metrics
    - Keep track of timing

    """
    clients: List[ClientRef] = []
    epoch_counter = 0
    client_data = {}

    def __init__(self, client_id_triple, num_epochs=3, config=None):
        log_rref = rpc.RRef(FLLogger())
        self.log_rref = log_rref
        self.num_epoch = num_epochs
        self.config = config
        self.tb_path = config.output_location
        self.ensure_path_exists(self.tb_path)
        self.tb_writer = SummaryWriter(f'{self.tb_path}/{config.experiment_prefix}_federator')
        self.create_clients(client_id_triple)
        self.config.init_logger(logging)
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.queue = []
        heapq.heapify(self.queue)
        self.optimizer = torch.optim.Adam(self.generator.parameters(),
                                          lr=self.args.get_learning_rate(), betas=(self.args.b1(), self.args.b2()))

    def create_clients(self, client_id_triple):
        for id, rank, world_size in client_id_triple:
            client = rpc.remote(id, Client, kwargs=dict(id=id, log_rref=self.log_rref,
                                                        rank=rank, world_size=world_size, config=self.config))
            writer = SummaryWriter(f'{self.tb_path}/{self.config.experiment_prefix}_client_{id}')
            self.clients.append(ClientRef(id, client, tensorboard_writer=writer))
            self.client_data[id] = []

    def gather_lbl_count(self, lbl_count):
        """
        This function gathers all labels counts from all workers at the server.
        Args:
            lbl_count: array of frequency of samples of each class at the current worker
        returns:
            workers_classes: array of arrays of labels counts of each class at the server
        """
        gather_list = [torch.zeros(len(lbl_count)) for _ in range(size)]
        res = [count_list.cpu().detach().numpy() for count_list in gather_list]
        return res

    def calculate_entropies(self, freqs):
        # TODO: get proper num_per_class
        num_per_class = []
        all_samples = sum(num_per_class)
        rat_per_class = [float(n / all_samples) for n in num_per_class]

        # Calculating entropy of each worker (on the server side) based on these frequencies
        self.entropies = [stats.entropy(np.array(freq_l) / sum(freq_l), rat_per_class) * (sum(freq_l) / all_samples)
                          for freq_l in freqs]

    def preprocess_groups(self, n=10, C=0.3):
        # TODO: fix label counts/frequencies and properly match clients into groups
        self.freqs = self.gather_lbl_count(lbl_count)
        self.calculate_entropies(self.freqs)
        self.groups = init_groups(n, C, self.freqs, self.clients)

    def select_clients(self, round):
        return balanced_sampling(self.groups, round)

    def init_dataloader(self, ):
        self.config.distributed = True
        self.dataset = self.config.DistDatasets[self.config.dataset_name](self.config)
        self.finished_init = True

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        logging.info('Done with init')

    def ping_all(self):
        for client in self.clients:
            logging.info(f'Sending ping to {client}')
            t_start = time.time()
            answer = _remote_method(Client.ping, client.ref)
            t_end = time.time()
            duration = (t_end - t_start)*1000
            logging.info(f'Ping to {client} is {duration:.3}ms')

    def rpc_test_all(self):
        for client in self.clients:
            res = _remote_method_async(Client.rpc_test, client.ref)
            while not res.done():
                pass

    def client_load_data(self):
        for client in self.clients:
            _remote_method_async(Client.init_dataloader, client.ref)

    def clients_ready(self):
        all_ready = False
        ready_clients = []
        while not all_ready:
            responses = []
            for client in self.clients:
                if client.name not in ready_clients:
                    responses.append((client, _remote_method_async(Client.is_ready, client.ref)))
            all_ready = True
            for res in responses:
                result = res[1].wait()
                if result:
                    logging.info(f'{res[0]} is ready')
                    ready_clients.append(res[0])
                else:
                    logging.info(f'Waiting for {res[0]}')
                    all_ready = False

            time.sleep(2)
        logging.info('All clients are ready')

    def remote_run_epoch(self, epochs, epoch):
        responses = []
        client_generators = []
        client_discriminators = []

        # broadcast generated datasets to clients and get trained discriminators back
        selected_clients = self.select_clients(epoch)
        for client in selected_clients:
            responses.append((client, _remote_method_async(Client.run_epochs, client.ref, epochs,
                                                           (self.generator, self.discriminator))))
        self.epoch_counter += epochs
        for res in responses:
            epoch_data, models = res[1].wait()
            self.client_data[epoch_data.client_id].append(epoch_data)
            logging.info(f'{res[0]} had a loss of {epoch_data.loss}')
            logging.info(f'{res[0]} had a epoch data of {epoch_data}')

            res[0].tb_writer.add_scalar('training loss',
                                        epoch_data.loss_train,  # for every 1000 minibatches
                                        self.epoch_counter * res[0].data_size)

            res[0].tb_writer.add_scalar('accuracy',
                                        epoch_data.accuracy,  # for every 1000 minibatches
                                        self.epoch_counter * res[0].data_size)

            client_generators.append(models)
            client_discriminators.append(models)

        selected_entropies = [self.entropies[idx] for idx in range(len(self.clients))
                              if self.clients[idx] in selected_clients]
        self.generator, self.discriminator = kl_weighting(self.generator, client_generators, selected_entropies), \
                                             kl_weighting(self.discriminator, client_discriminators, selected_entropies)

    def update_client_data_sizes(self):
        responses = []
        for client in self.clients:
            responses.append((client, _remote_method_async(Client.get_client_datasize, client.ref)))
        for res in responses:
            res[0].data_size = res[1].wait()
            logging.info(f'{res[0]} had a result of datasize={res[0].data_size}')

    def remote_test_sync(self):
        responses = []
        for client in self.clients:
            responses.append((client, _remote_method_async(Client.test, client.ref)))

        for res in responses:
            accuracy, loss, class_precision, class_recall = res[1].wait()
            logging.info(f'{res[0]} had a result of accuracy={accuracy}')

    def save_epoch_data(self):
        file_output = f'./{self.config.output_location}'
        self.ensure_path_exists(file_output)
        for key in self.client_data:
            filename = f'{file_output}/{key}_epochs.csv'
            logging.info(f'Saving data at {filename}')
            with open(filename, "w") as f:
                w = DataclassWriter(f, self.client_data[key], EpochData)
                w.write()

    def ensure_path_exists(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def run(self):
        """
        Main loop of the Federator
        :return:
        """
        # # Make sure the clients have loaded all the data
        self.client_load_data()
        self.ping_all()
        self.clients_ready()
        self.update_client_data_sizes()

        self.init_dataloader()
        self.preprocess_groups()

        epoch_to_run = self.num_epoch
        addition = 0
        epoch_to_run = self.config.epochs
        epoch_size = self.config.epochs_per_cycle
        for epoch in range(epoch_to_run):
            print(f'Running epoch {epoch}')
            self.remote_run_epoch(epoch_size, epoch)
            addition += 1
        logging.info('Printing client data')
        print(self.client_data)

        logging.info(f'Saving data')
        self.save_epoch_data()
        logging.info(f'Federator is stopping')


if __name__ == '__main__':
    world_size = 1
    Federator([(f"client{r}", r, world_size) for r in range(1, world_size)]).run()
