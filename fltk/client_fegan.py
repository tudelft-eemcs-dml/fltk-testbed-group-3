import datetime
import random
from typing import List

from ot import wasserstein_1d
from torch.distributed import rpc
from torch.autograd import Variable
import logging
import numpy as np
import gc
from fltk.client import Client
from fltk.util.wassdistance import SinkhornDistance
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
        self.batch_size = 300
        self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)

    def return_distribution(self):
        labels = self.dataset.load_train_dataset()[1]
        unique, counts = np.unique(labels, return_counts=True)
        return sorted((zip(unique, counts)))

    def weight_scale(self, m):
        if type(m) == torch.nn.Conv2d:
            m.weight = torch.nn.Parameter((m.weight * 0.02) / torch.max(m.weight, dim=1, keepdim=True)[0] - 0.01)

    def init_dataloader(self, ):
        self.args.distributed = True
        self.args.rank = self.rank
        self.args.world_size = self.world_size
        self.dataset = self.args.DistDatasets[self.args.dataset_name](self.args)
        self.inputs, lbl = self.dataset.load_train_dataset()
        del lbl
        self.finished_init = True

        self.batch_size = 300

        logging.info('Done with init')

    def J_generator(self, disc_out):
        return (1 / self.batch_size) * torch.sum(torch.log(torch.clamp(1 - disc_out, min=self.epsilon)))

    def A_hat(self, disc_out):
        return (1 / self.batch_size) * torch.sum(torch.log(torch.clamp(disc_out, min=self.epsilon)))

    def B_hat(self, disc_out):
        return (1 / self.batch_size) * torch.sum(torch.log(torch.clamp(1 - disc_out, min=self.epsilon)))

    def wasserstein_loss(self, y_true, y_pred):
        return torch.mean(y_true * y_pred)

    def train_fe(self, epoch, net):
        generator, discriminator = net
        # generator.zero_grad()
        # discriminator.zero_grad()
        optimizer_generator = torch.optim.Adam(generator.parameters(),
                                               lr=0.0002,
                                               betas=(0.5, 0.999))
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(),
                                                   lr=0.0002,
                                                   betas=(0.5, 0.999))

        if self.args.distributed:
            self.dataset.train_sampler.set_epoch(epoch)

        for batch_idx in range(len(self.inputs) // self.batch_size + 1):

            try:
                inputs = torch.from_numpy(self.inputs[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size])
            except:
                inputs = torch.from_numpy(self.inputs[batch_idx * self.batch_size:])

            # inputs = torch.from_numpy(self.inputs[[random.randrange(self.inputs.shape[0]) for _ in range(self.batch_size)]])

            optimizer_generator.zero_grad()
            optimizer_discriminator.zero_grad()

            # apply weight scale for WGAN loss implementation
            # with torch.no_grad():
            #     discriminator = discriminator.apply(self.weight_scale)

            noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))), requires_grad=False)
            generated_imgs = generator(noise.detach())
            d_generator = discriminator(generated_imgs)

            # minmax loss
            generator_loss = self.J_generator(d_generator)

            # wasserstein distance loss
            # generator_loss = torch.tensor([wasserstein_1d(torch.flatten(d_generator).detach().numpy(),
            #                                               torch.ones(self.batch_size).detach().numpy())], requires_grad=True)

            # wasserstein distance approximation via sinkhorn
            # generator_loss, _, _ = self.sinkhorn(d_generator, torch.ones(self.batch_size, 1))

            # wassersetin loss from wgan
            # generator_loss = self.wasserstein_loss(d_generator, (-1.0) * torch.ones(self.batch_size))
            # generator_loss.require_grad = True
            generator_loss.backward(retain_graph=True)

            # minmax loss
            fake_loss = self.B_hat(d_generator.detach())
            real_loss = self.A_hat(discriminator(inputs))

            discriminator_loss = 0.5 * (real_loss + fake_loss)

            # wasserstein distance
            # fake_loss = wasserstein_1d(torch.flatten(d_generator).detach().numpy(),
            #                                   torch.zeros(self.batch_size).detach().numpy())
            # real_loss = wasserstein_1d(torch.flatten(discriminator(inputs)).detach().numpy(),
            #                                   torch.ones(self.batch_size).detach().numpy())

            # wassersetin distance via sinkhorn
            # fake_loss, _, _ = self.sinkhorn(d_generator, torch.zeros(self.batch_size, 1))
            # real_loss, _, _ = self.sinkhorn(discriminator(inputs), torch.ones(self.batch_size, 1))

            # wasserstein loss
            # generated_imgs = generator(noise.detach())
            # d_generator = discriminator(generated_imgs.detach())
            # fake_loss = self.wasserstein_loss(d_generator.detach(),
            #                                   (-1) * torch.ones(self.batch_size))
            # real_loss = self.wasserstein_loss(discriminator(inputs),
            #                                   torch.ones(self.batch_size))
            # discriminator_loss = real_loss + fake_loss

            discriminator_loss.backward()
            generator_loss.backward()

            optimizer_generator.step()
            optimizer_discriminator.step()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())

        del noise, generated_imgs, fake_loss, real_loss, discriminator_loss, generator_loss, \
            optimizer_discriminator, optimizer_generator, d_generator, inputs
        gc.collect()

        return generator, discriminator

    def run_epochs(self, num_epoch, net=None):
        start_time_train = datetime.datetime.now()

        for e in range(num_epoch):
            net = self.train_fe(self.epoch_counter, net)
            self.epoch_counter += 1
            gc.collect()
        elapsed_time_train = datetime.datetime.now() - start_time_train
        train_time_ms = int(elapsed_time_train.total_seconds() * 1000)

        data = FeGANEpochData(self.epoch_counter, train_time_ms, 0, net, client_id=self.id)
        # self.epoch_results.append(data)

        return data
