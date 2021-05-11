import copy
import datetime
import os
import random
import time
from dataclasses import dataclass
from typing import List

import torch
from torch.distributed import rpc
from torch.autograd import Variable
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from fltk.client import Client
from fltk.schedulers import MinCapableStepLR
from fltk.util.arguments import Arguments
from fltk.util.fed_avg import average_nn_parameters
from fltk.util.weight_init import *
from fltk.util.log import FLLogger
from fltk.nets.md_gan import *

import yaml

from fltk.util.results import EpochData

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


class ClientMDGAN(Client):
    counter = 0
    finished_init = False
    dataset = None
    epoch_results: List[EpochData] = []
    epoch_counter = 0

    def __init__(self, id, log_rref, rank, world_size, config=None):
        super().__init__(id, log_rref, rank, world_size, config)
        logging.info(f'Welcome to MD client {id}')
        self.discriminator = Discriminator()

    def init_dataloader(self, ):
        self.args.distributed = True
        self.args.rank = self.rank
        self.args.world_size = self.world_size
        # self.dataset = DistCIFAR10Dataset(self.args)
        self.dataset = self.args.DistDatasets[self.args.dataset_name](self.args)
        self.finished_init = True

        self.discriminator.apply(weights_init_normal)
        logging.info('Done with init')

    def train(self, epoch, Xs=None):
        """
        :param epoch: Current epoch #
        :type epoch: int
        """
        if Xs:
            X_d, X_g = Xs

            # save model
            if self.args.should_save_model(epoch):
                self.save_model(epoch, self.args.get_epoch_save_start_suffix())

            running_loss = 0.0
            final_running_loss = 0.0
            if self.args.distributed:
                self.dataset.train_sampler.set_epoch(epoch)

            for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 1):
                inputs, labels, fake = inputs.to(self.device), labels.to(self.device), torch.zeros(inputs.shape[0])

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Sample noise as generator input
                noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (inputs.shape[0], 100))))

                outputs = self.discriminator(inputs)

                real_loss = self.loss_function(outputs, labels)
                fake_loss = self.loss_function(self.discriminator(X_d.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)
                d_loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += d_loss.item()
                if i % self.args.get_log_interval() == 0:
                    self.args.get_logger().info('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / self.args.get_log_interval()))
                    final_running_loss = running_loss / self.args.get_log_interval()
                    running_loss = 0.0

            self.scheduler.step()

            # save model
            if self.args.should_save_model(epoch):
                self.save_model(epoch, self.args.get_epoch_save_end_suffix())

            return final_running_loss, self.get_nn_parameters()

    def test(self):
        self.net.eval()

        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        accuracy = 100 * correct / total
        confusion_mat = confusion_matrix(targets_, pred_)

        class_precision = self.calculate_class_precision(confusion_mat)
        class_recall = self.calculate_class_recall(confusion_mat)

        self.args.get_logger().debug('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))
        self.args.get_logger().debug('Test set: Loss: {}'.format(loss))
        self.args.get_logger().debug("Classification Report:\n" + classification_report(targets_, pred_))
        self.args.get_logger().debug("Confusion Matrix:\n" + str(confusion_mat))
        self.args.get_logger().debug("Class precision: {}".format(str(class_precision)))
        self.args.get_logger().debug("Class recall: {}".format(str(class_recall)))

        return accuracy, loss, class_precision, class_recall

    def run_epochs(self, num_epoch, Xs=None):
        start_time_train = datetime.datetime.now()
        loss = weights = None

        for e in range(num_epoch):
            loss, weights = self.train(self.epoch_counter, Xs)
            self.epoch_counter += 1
        elapsed_time_train = datetime.datetime.now() - start_time_train
        train_time_ms = int(elapsed_time_train.total_seconds()*1000)

        start_time_test = datetime.datetime.now()
        accuracy, test_loss, class_precision, class_recall = self.test()
        elapsed_time_test = datetime.datetime.now() - start_time_test
        test_time_ms = int(elapsed_time_test.total_seconds()*1000)

        data = EpochData(self.epoch_counter, train_time_ms, test_time_ms, loss,
                         accuracy, test_loss, class_precision, class_recall, client_id=self.id)
        self.epoch_results.append(data)

        # Copy GPU tensors to CPU
        for k, v in weights.items():
            weights[k] = v.cpu()
        return data, weights
