import math
from collections import Counter
from queue import Queue
import json

from pandas import np
from scipy import stats
from torch.autograd import Variable

from fltk.client_fegan import _remote_method_async, ClientFeGAN
from fltk.federator import *
from fltk.strategy.client_selection import balanced_sampling
from fltk.util.fed_avg import kl_weighting
from fltk.nets.ls_gan import *
from fltk.util.fid_score import calculate_activation_statistics, calculate_frechet_distance
from fltk.util.inception import InceptionV3
from fltk.util.weight_init import *
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG)


class FederatorFeGAN(Federator):

    def __init__(self, client_id_triple, num_epochs=3, config=None):
        super().__init__(client_id_triple, num_epochs, config, ClientFeGAN)
        self.groups = []
        self.generator = Generator()
        self.generator.apply(weights_init_normal)
        self.discriminator = Discriminator()
        self.discriminator.apply(weights_init_normal)
        self.latent_dim = 10
        self.batch_size = 1000
        self.fids = []
        self.inceptions = []
        self.fic_model = InceptionV3()

    def gather_class_distributions(self):
        responses = []
        self.class_distributions = []
        for client in self.clients:
            responses.append((client, _remote_method_async(
                ClientFeGAN.return_distribution, client.ref)))
        for res in responses:
            dist = res[1].wait()
            self.class_distributions.append(dist)

    def preprocess_groups(self, c=0.3):
        self.gather_class_distributions()
        num_per_class = Counter({})
        for dist in self.class_distributions:
            num_per_class += Counter(dict(dist))

        num_per_class = dict(num_per_class).values()
        self.init_groups(c, len(num_per_class))

        # we need entropies for client weighting during kl-divergence calculations
        # idea from code, not from official paper...
        all_samples = sum(num_per_class)
        rat_per_class = [float(n / all_samples) for n in num_per_class]
        cleaned_distributions = [[c for _, c in dist]
                                 for dist in self.class_distributions]
        self.entropies = [stats.entropy(np.array(freq_l)/sum(freq_l), rat_per_class) * (sum(freq_l) / all_samples)
                          for freq_l in cleaned_distributions]

    def init_dataloader(self, ):
        self.config.distributed = True
        self.dataset = self.config.DistDatasets[self.config.dataset_name](
            self.config)
        self.test_imgs, lbls = self.dataset.load_test_dataset()
        self.finished_init = True

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        del lbls
        logging.info('Done with init')

    # TODO: cleanup
    def init_groups(self, c, label_size):
        gp_size = math.ceil(c * len(self.clients))
        done = False
        size = len(self.clients)

        wrk_cls = [[False for i in range(label_size)] for j in range(size)]
        cls_q = [Queue(maxsize=size) for _ in range(10)]
        for i, cls_list in enumerate(self.class_distributions):
            wrk_cls[i] = [True if freq != 0 else False for _, freq in cls_list]
        for worker, class_list in enumerate(reversed(wrk_cls)):
            for cls, exist in enumerate(class_list):
                if exist:
                    cls_q[cls].put(size - worker - 1)
        taken_count = [0 for _ in range(label_size)]

        print('generating balanced groups for training...')
        while not done:
            visited = [False for _ in range(size)]
            g = []
            for _ in range(gp_size):
                cls = np.where(taken_count == np.amin(taken_count))[0][0]
                assert 0 <= cls <= len(taken_count)
                done_q = False
                count = 0
                while not done_q:
                    wrkr = cls_q[cls].get()
                    if not visited[wrkr] and wrk_cls[wrkr][cls]:
                        g.append(wrkr)
                        taken_count += self.class_distributions[wrkr][1]
                        visited[wrkr] = True
                        done_q = True
                    cls_q[cls].put(wrkr)
                    count += 1
                    if count == size:
                        done_q = True

            self.groups.append(g)
            # TODO: should be hyperparam
            if len(self.groups) > 10:
                done = True

    def test(self, fl_round):
        with torch.no_grad():
            file_output = f'./{self.config.output_location}'
            self.ensure_path_exists(file_output)
            fid_z = Variable(torch.FloatTensor(np.random.normal(
                0, 1, (self.test_imgs.shape[0], self.latent_dim))))
            gen_imgs = self.generator(fid_z.detach())
            mu_gen, sigma_gen = calculate_activation_statistics(
                gen_imgs, self.fic_model)
            mu_test, sigma_test = calculate_activation_statistics(
                torch.from_numpy(self.test_imgs), self.fic_model)
            fid = calculate_frechet_distance(
                mu_gen, sigma_gen, mu_test, sigma_test)
            print("FL-round {} FID Score: {}, IS Score: {}".format(fl_round, fid, mu_gen))

            self.fids.append(fid)
            # self.inceptions.append(mu_gen)

    def checkpoint(self, fl_round):
        # For fault tolerance
        file_output = f'./{self.config.output_location}'
        self.ensure_path_exists(file_output)
        print("saving checkpoint...")
        state = {'disc': self.discriminator.state_dict(), 'gen': self.generator.state_dict(), 'epoch': fl_round,
                 'fl_round': fl_round}
        torch.save(state, file_output + "/checkpoint")

    def plot_score_data(self):
        print(self.fids)
        file_output = f'./{self.config.output_location}'
        self.ensure_path_exists(file_output)

        plt.plot(range(0, self.config.epochs, 25), self.fids, 'b')
        # plt.plot(range(self.config.epochs), self.inceptions, 'r')
        plt.xlabel('Federator runs')
        plt.ylabel('FID')

        filename = f'{file_output}/{self.config.epochs}_epochs_fe_gan_wd2.png'
        logging.info(f'Saving data at {filename}')

        plt.savefig(filename)

        json_data = {"fids": self.fids}
        with open(f'{file_output}/fid_{self.config.epochs}_epochs_fe_gan.json', 'w') as fp:
            json.dump(json_data, fp)

    def remote_run_epoch(self, epochs, epoch=None):
        responses = []
        client_generators = []
        client_discriminators = []

        # broadcast generator and discriminator to clients
        selected_clients = balanced_sampling(self.clients, self.groups, epoch)
        for client in selected_clients:
            responses.append((client, _remote_method_async(ClientFeGAN.run_epochs, client.ref, epochs,
                                                           (self.generator, self.discriminator))))
        self.epoch_counter += epochs
        for res in responses:
            epoch_data = res[1].wait()
            # self.client_data[epoch_data.client_id].append(epoch_data)
            logging.info(f'{res[0]} had a epoch data of {epoch_data}')

            client_generators.append(epoch_data.net[0])
            client_discriminators.append(epoch_data.net[1])

        selected_entropies = [self.entropies[idx] for idx in range(len(self.clients))
                              if self.clients[idx] in selected_clients]
        self.generator, self.discriminator = kl_weighting(self.generator, client_generators, selected_entropies), \
            kl_weighting(self.discriminator,
                         client_discriminators, selected_entropies)

        if epoch % 25 == 0:
            self.test(epoch)

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
        start_time_train = datetime.datetime.now()
        for epoch in range(epoch_to_run):
            epoch_start_time = datetime.datetime.now()
            print(f'Running epoch {epoch}')
            self.remote_run_epoch(epoch_size, epoch)
            addition += 1
            self.plot_score_data()
            elapsed_time_epoch = datetime.datetime.now() - epoch_start_time
            self.epoch_times.append(elapsed_time_epoch.total_seconds())

            self.plot_time_data(gan="fe")

        self.plot_time_data(gan="fe")

        elapsed_time_train = datetime.datetime.now() - start_time_train
        train_time_ms = int(elapsed_time_train.total_seconds() * 1000)
        logging.info('Printing client data')
        print(self.client_data)

        logging.info(f'Federator is stopping')
        self.plot_score_data()

        throughput = round(train_time_ms / epoch_to_run, 2)
        print('Throughput: {} ms'.format(throughput))
