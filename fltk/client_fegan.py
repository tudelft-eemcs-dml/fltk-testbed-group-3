import datetime
import random
from typing import List

from torch.distributed import rpc
from torch.autograd import Variable
import logging
import numpy as np

from fltk.client import Client
from fltk.util.weight_init import *

from fltk.util.results import EpochData, FeGANEpochData

logging.basicConfig(level=logging.DEBUG)


def _call_method(method, rref, *args, **kwargs):
    """helper for _remote_method()"""
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    """
    executes method(*args, **kwargs) on the from the machine that owns rref

    very similar to rref.remote().method(*args, **kwargs), but method() doesn't have to be in the remote scope
    """
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


def _remote_method_async(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)


def average_models(model):
    return model


class ClientFeGAN(Client):
    counter = 0
    finished_init = False
    dataset = None
    epoch_results: List[EpochData] = []
    epoch_counter = 0
    epsilon = 0.00000001

    def __init__(self, id, log_rref, rank, world_size, config=None):
        super().__init__(id, log_rref, rank, world_size, config)
        logging.info(f'Welcome to FE client {id}')
        self.latent_dim = 10
        self.batch_size = 100

    def return_distribution(self):
        labels = self.dataset.load_train_dataset()[1]
        unique, counts = np.unique(labels, return_counts=True)
        return sorted((zip(unique, counts)))

    def init_dataloader(self, ):
        self.args.distributed = True
        self.args.rank = self.rank
        self.args.world_size = self.world_size
        self.dataset = self.args.DistDatasets[self.args.dataset_name](self.args)
        self.inputs, lbl = self.dataset.load_train_dataset()
        del lbl
        self.finished_init = True

        self.batch_size = 100

        logging.info('Done with init')

    def J_generator(self, disc_out):
        return (1 / self.batch_size) * torch.sum(torch.log(torch.clamp(1 - disc_out, min=self.epsilon)))

    def A_hat(self, disc_out):
        return (1 / self.batch_size) * torch.sum(torch.log(torch.clamp(disc_out, min=self.epsilon)))

    def B_hat(self, disc_out):
        return (1 / self.batch_size) * torch.sum(torch.log(torch.clamp(1 - disc_out, min=self.epsilon)))

    def train_fe(self, epoch, net):
        generator, discriminator = net
        generator.zero_grad()
        discriminator.zero_grad()
        optimizer_generator = torch.optim.Adam(generator.parameters(),
                                               lr=0.0002,
                                               betas=(0.5, 0.999))
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(),
                                                   lr=0.0002,
                                                   betas=(0.5, 0.999))

        if self.args.distributed:
            self.dataset.train_sampler.set_epoch(epoch)

        inputs = torch.from_numpy(self.inputs[[random.randrange(self.inputs.shape[0]) for _ in range(self.batch_size)]]).detach()

        optimizer_generator.zero_grad()
        optimizer_discriminator.zero_grad()

        noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))
        generated_imgs = generator(noise)
        d_generator = discriminator(generated_imgs)
        generator_loss = self.J_generator(d_generator)
        generator_loss.require_grad = True
        generator_loss.backward(retain_graph=True)

        fake_loss = self.B_hat(d_generator)
        real_loss = self.A_hat(discriminator(inputs))
        discriminator_loss = 0.5 * (real_loss + fake_loss)
        discriminator_loss.require_grad = True
        discriminator_loss.backward()

        optimizer_generator.step()
        optimizer_discriminator.step()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())

        return generator, discriminator

    def run_epochs(self, num_epoch, net=None):
        start_time_train = datetime.datetime.now()
        gen, disc = None, None

        for e in range(num_epoch):
            gen, disc = self.train_fe(self.epoch_counter, net)
            self.epoch_counter += 1
        elapsed_time_train = datetime.datetime.now() - start_time_train
        train_time_ms = int(elapsed_time_train.total_seconds() * 1000)

        data = FeGANEpochData(self.epoch_counter, train_time_ms, 0, (gen, disc), client_id=self.id)
        # self.epoch_results.append(data)

        return data
