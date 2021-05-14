import copy
import math

from pandas import np
from torch.autograd import Variable
from torch.distributed import rpc

from fltk.client import _remote_method
from fltk.federator import *
from fltk.client_mdgan import ClientMDGAN, _remote_method_async
from fltk.strategy.client_selection import random_selection
from fltk.util.fed_avg import average_nn_parameters
from fltk.util.fid_score import calculate_activation_statistics, calculate_frechet_distance
from fltk.util.inception import InceptionV3
from fltk.util.log import FLLogger
from fltk.nets.md_gan import *
from fltk.util.weight_init import *
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging

from fltk.util.results import EpochData

logging.basicConfig(level=logging.DEBUG)


class FederatorMDGAN(Federator):
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

    # TODO: add lr/bs/latdim to args
    def __init__(self, client_id_triple, num_epochs=3, config=None):
        super().__init__(client_id_triple, num_epochs, config, ClientMDGAN)
        self.init_dataloader()

        self.generator = Generator(32)
        self.generator.apply(weights_init_normal)
        self.optimizer = torch.optim.Adam(self.generator.parameters(),
                                          lr=0.0002,
                                          betas=(0.5, 0.999))
        self.latent_dim = 10
        # K <= N
        self.k = len(client_id_triple) - 1
        # TODO this is merely wrong - federator cannot access whole dataset
        self.batch_size = math.floor(self.dataset[0].shape[0] / self.k)
        self.introduce_clients()

    def introduce_clients(self):
        ref_clients = []
        responses = []
        for client in self.clients:
            ref_clients.append((client.ref, client.name))
        for client in self.clients:
            responses.append(_remote_method_async(ClientMDGAN.get_client_list, client.ref, client_list=ref_clients))

        for res in responses:
            res.wait()

    def init_dataloader(self, ):
        self.config.distributed = True
        self.dataset = self.config.DistDatasets[self.config.dataset_name](self.config).load_train_dataset()
        self.testset = self.config.DistDatasets[self.config.dataset_name](self.config).load_test_dataset()
        self.finished_init = True

        logging.info('Done with init')

    def ping_all(self):
        for client in self.clients:
            logging.info(f'Sending ping to {client}')
            t_start = time.time()
            answer = _remote_method(ClientMDGAN.ping, client.ref)
            t_end = time.time()
            duration = (t_end - t_start)*1000
            logging.info(f'Ping to {client} is {duration:.3}ms')

    def rpc_test_all(self):
        for client in self.clients:
            res = _remote_method_async(ClientMDGAN.rpc_test, client.ref)
            while not res.done():
                pass

    def client_load_data(self):
        for client in self.clients:
            _remote_method_async(ClientMDGAN.init_dataloader, client.ref)
            _remote_method_async(ClientMDGAN.set_batch_size, client.ref, batch_size=self.batch_size)

    def test_generator(self, fl_round):
        eval_samples = 50
        fic_model = InceptionV3()
        test_imgs = self.testset[0]
        fid_z = Variable(torch.Tensor(np.random.normal(0, 1, (eval_samples, self.latent_dim))))
        gen_imgs = self.generator(fid_z)
        mu_gen, sigma_gen = calculate_activation_statistics(gen_imgs, fic_model)
        mu_test, sigma_test = calculate_activation_statistics(test_imgs[:eval_samples], fic_model)
        fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_test, sigma_test)
        print("FL-round {} FID Score: {}".format(fl_round, fid))

    def remote_run_epoch(self, epochs, fl_round=0):
        responses = []
        client_errors = []

        self.optimizer.zero_grad()

        # broadcast generated datasets to clients and get trained discriminators back
        selected_clients = self.select_clients(self.config.clients_per_round)

        for client in selected_clients:
            # Get some real conformations from the train data, not sure why we need it, not in pseudocode...
            real = self.dataset[epochs * self.batch_size:(epochs + 1) * self.batch_size]
            # Sample noise as generator input and feed to clients based on their datat size
            noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))
            X_g = self.generator(noise)

            noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))
            X_d = self.generator(noise)
            responses.append((client, _remote_method_async(ClientMDGAN.run_epochs, client.ref,
                                                           epochs, fl_round, (X_d, X_g))))

        self.epoch_counter += epochs
        for res in responses:
            epoch_data = res[1].wait()
            self.client_data[epoch_data.client_id].append(epoch_data)
            logging.info(f'{res[0]} had a loss of {epoch_data.loss}')
            logging.info(f'{res[0]} had a epoch data of {epoch_data}')

            res[0].tb_writer.add_scalar('training loss',
                                        epoch_data.loss_train,  # for every 1000 minibatches
                                        self.epoch_counter * res[0].data_size)

            res[0].tb_writer.add_scalar('accuracy',
                                        epoch_data.accuracy,  # for every 1000 minibatches
                                        self.epoch_counter * res[0].data_size)

            client_errors.append(epoch_data.loss)

        # TODO: not sure this is actually correct

        client_errors = torch.tensor(client_errors, dtype=torch.float)
        client_errors.requires_grad = True

        target = torch.zeros(len(client_errors), dtype=torch.float)
        target.requires_grad = True

        g_loss = torch.nn.L1Loss().forward(client_errors, target)
        g_loss.backward()
        self.optimizer.step()

        logging.info('Gradient is updated')
        self.test_generator(fl_round)

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
