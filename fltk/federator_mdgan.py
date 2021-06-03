import copy
import math
import random

from pandas import np
from torch.autograd import Variable

from fltk.client import _remote_method
from fltk.federator import *
from fltk.client_mdgan import ClientMDGAN, _remote_method_async
from fltk.util.fid_score import calculate_activation_statistics, calculate_frechet_distance
from fltk.util.inception import InceptionV3
from fltk.nets.ls_gan import *
from fltk.util.weight_init import *
import logging
import matplotlib.pyplot as plt
from torch.distributed.optim import DistributedOptimizer
from torch import optim

logging.basicConfig(level=logging.DEBUG)


class FederatorMDGAN(Federator):

    def __init__(self, client_id_triple, num_epochs=3, config=None):
        super().__init__(client_id_triple, num_epochs, config, ClientMDGAN)
        self.init_dataloader()

        self.generator = Generator(32)
        self.generator.apply(weights_init_normal)
        # self.optimizer = torch.optim.Adam(self.generator.parameters(),
        #                                   lr=0.0002,
        #                                   betas=(0.5, 0.999))
        self.optimizer = torch.distributed.optim.DistributedOptimizer(
            optim.Adam,  self.param_rrefs(self.generator), lr=2e-4, betas=(0.5, 0.99))
        self.latent_dim = 10
        # K <= N
        self.k = len(client_id_triple) - 1
        self.batch_size = math.floor(self.dataset[0].shape[0] / self.k) // 10
        self.introduce_clients()
        self.fids = []
        self.discriminator = Discriminator(32)
        self.inceptions = []
        self.epsilon = 0.00000001
        self.fic_model = InceptionV3()

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
        with torch.no_grad():
            file_output = f'./{self.config.output_location}'
            self.ensure_path_exists(file_output)
            test_imgs = self.testset[0]
            fid_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (test_imgs.shape[0], self.latent_dim))))
            gen_imgs = self.generator(fid_z.detach())
            mu_gen, sigma_gen = calculate_activation_statistics(gen_imgs, self.fic_model)
            mu_test, sigma_test = calculate_activation_statistics(torch.from_numpy(test_imgs), self.fic_model)
            fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_test, sigma_test)
            print("FL-round {} FID Score: {}, IS Score: {}".format(fl_round, fid, mu_gen))

            self.fids.append(fid)
            # self.inceptions.append(mu_gen)

    def param_rrefs(self, module):
        """grabs remote references to the parameters of a module"""
        param_rrefs = []
        for param in module.parameters():
            param_rrefs.append(rpc.RRef(param))
        print(param_rrefs)
        return param_rrefs

    def J_generator(self, noise, discriminator):
        return (1 / self.batch_size) * torch.sum(torch.log(torch.clamp(1 - discriminator(noise), min=self.epsilon)))

    def remote_run_epoch(self, epochs, fl_round=0):
        responses = []
        client_rrefs = []

        # broadcast generated datasets to clients and get trained discriminators back
        selected_clients = self.select_clients(self.config.clients_per_round)

        # X_g = []
        # X_d = []
        # for i in range(self.k):
        #     noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))
        #     X_g.append(self.generator(noise.detach()))
        #
        #     noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))
        #     X_d.append(self.generator(noise.detach()))
        #
        # samples_d = [random.randrange(self.k) for _ in range(len(selected_clients))]
        # samples_g = [random.randrange(self.k) for _ in range(len(selected_clients))]

        for id, client in enumerate(selected_clients):
            # Sample noise as generator input and feed to clients based on their datat size
            # X_d_i = X_d[samples_d[id]]
            # X_g_i = X_g[samples_g[id]]
            noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))
            X_g_i = self.generator(noise)

            noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))
            X_d_i = self.generator(noise)

            responses.append((client, _remote_method_async(ClientMDGAN.run_epochs, client.ref,
                                                           epochs, fl_round, (X_d_i, X_g_i))))
            client_rrefs.append(client.ref)

        self.epoch_counter += epochs
        for res in responses:
            epoch_data = res[1].wait()

        with torch.distributed.autograd.context() as G_context:
            loss_g_list = []
            for client_rref in client_rrefs:
                print("G step update")
                loss_g_list.append(client_rref.rpc_async().loss_generator())
            loss_accumulated = loss_g_list[0].wait()
            for j in range(0, len(loss_g_list)):
                loss_accumulated += loss_g_list[j].wait()

            torch.distributed.autograd.backward(G_context, [loss_accumulated])
            self.optimizer.step(G_context)

        logging.info('Gradient is updated')
        if fl_round % 25 == 0:
            self.test_generator(fl_round)

    def plot_score_data(self):
        file_output = f'./{self.config.output_location}'
        self.ensure_path_exists(file_output)

        plt.plot(range(0, self.config.epochs, 25), self.fids, 'b')
        # plt.plot(range(self.config.epochs), self.inceptions, 'r')
        plt.xlabel('Federator runs')
        plt.ylabel('FID')

        filename = f'{file_output}/fid_{self.config.epochs}_epochs_md_gan.png'
        logging.info(f'Saving data at {filename}')

        plt.savefig(filename)

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
        start_time_train = datetime.datetime.now()
        for epoch in range(epoch_to_run):
            print(f'Running epoch {epoch}')
            self.remote_run_epoch(epoch_size, epoch)
            addition += 1
        elapsed_time_train = datetime.datetime.now() - start_time_train
        train_time_ms = int(elapsed_time_train.total_seconds() * 1000)
        logging.info('Printing client data')
        # print(self.client_data)

        logging.info(f'Federator is stopping')
        self.plot_score_data()

        throughput = round(train_time_ms / epoch_to_run, 2)
        print('Throughput: {} ms'.format(throughput))
