from pandas import np
from torch.autograd import Variable
from torch.distributed import rpc

from fltk.client import _remote_method
from fltk.federator import *
from fltk.client_mdgan import ClientMDGAN, _remote_method_async
from fltk.strategy.client_selection import random_selection
from fltk.util.fed_avg import average_nn_parameters
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

    def __init__(self, client_id_triple, num_epochs=3, config=None):
        super().__init__(client_id_triple, num_epochs, config)
        self.generator = Generator()
        self.optimizer = torch.optim.Adam(self.generator.parameters(),
                                          lr=self.args.get_learning_rate(),
                                          betas=(self.args.b1(), self.args.b2()))

    def init_dataloader(self, ):
        self.config.distributed = True
        self.dataset = self.config.DistDatasets[self.config.dataset_name](self.config)
        self.finished_init = True

        self.generator.apply(weights_init_normal)
        logging.info('Done with init')

    def remote_run_epoch(self, epochs):
        responses = []
        client_weights = []

        imgs = self.dataset.get_train_loader()

        noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], 100))))
        # Generate a batch of images
        X_g = self.generator(noise)

        fake = X_g, torch.zeros(imgs.shape[0])
        valid = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)

        self.optimizer.zero_grad()

        # Sample noise as generator input
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], 100))))
        X_g = self.generator(z)

        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], 100))))
        X_d = self.generator(z)

        # broadcast generated datasets to clients and get trained discriminators back
        selected_clients = self.select_clients(self.config.clients_per_round)
        for client in selected_clients:
            responses.append((client, _remote_method_async(Client.run_epochs, client.ref, epochs, (X_d, X_g))))
        self.epoch_counter += epochs
        for res in responses:
            epoch_data, weights = res[1].wait()
            self.client_data[epoch_data.client_id].append(epoch_data)
            logging.info(f'{res[0]} had a loss of {epoch_data.loss}')
            logging.info(f'{res[0]} had a epoch data of {epoch_data}')

            res[0].tb_writer.add_scalar('training loss',
                                        epoch_data.loss_train,  # for every 1000 minibatches
                                        self.epoch_counter * res[0].data_size)

            res[0].tb_writer.add_scalar('accuracy',
                                        epoch_data.accuracy,  # for every 1000 minibatches
                                        self.epoch_counter * res[0].data_size)

            client_weights.append(weights)

        # average discriminators from clients and update own generator
        updated_model = average_nn_parameters(client_weights)
        discriminator = Discriminator()
        discriminator.parameters = updated_model

        self.optimizer.zero_grad()
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], 100))))
        X_g = self.generator(z)
        g_loss = torch.nn.MSELoss(discriminator(X_g), valid)
        g_loss.backward()
        self.optimizer.step()

        responses = []
        for client in self.clients:
            responses.append(
                (client, _remote_method_async(Client.update_nn_parameters, client.ref, new_params=updated_model)))

        for res in responses:
            res[1].wait()
        logging.info('Weights are updated')

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
            self.remote_run_epoch(epoch_size)
            addition += 1
        logging.info('Printing client data')
        print(self.client_data)

        logging.info(f'Saving data')
        self.save_epoch_data()
        logging.info(f'Federator is stopping')


if __name__ == '__main__':
    world_size = 1
    FederatorMDGAN([(f"client{r}", r, world_size) for r in range(1, world_size)]).run()
