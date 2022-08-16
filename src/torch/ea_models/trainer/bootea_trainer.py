import copy
import gc
import math
import queue
import random
import time
from collections import OrderedDict

import numpy as np
import ray
from joblib._multiprocessing_helpers import mp
from torch.autograd import Variable
import torch
import torch.nn as nn
import pandas as pd

from src.py.base.optimizers import get_optimizer_torch
from src.py.evaluation.alignment import task_divide
from src.py.load import batch
from src.py.load.kg import KG
from src.py.util.util import to_tensor
from src.torch.ea_models.models.bootea import BootEA, generate_supervised_triples, generate_pos_batch, \
    calculate_likelihood_mat, bootstrapping
from src.torch.kge_models.basic_model import align_model_trainer


def early_stop(flag1, flag2, flag):
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False


class bootea_trainer(align_model_trainer):
    def __init__(self):
        super(bootea_trainer, self).__init__()
        self.labeled_align = None
        self.ref_ent2 = None
        self.ref_ent1 = None
        self.training_batch_queue = None
        self.args = None
        self.kgs = None
        self.model = None
        self.neighbors1, self.neighbors2 = None, None
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = None
        self.optimizer = None

    def init(self, args, kgs):
        self.args = args
        self.kgs = kgs
        self.labeled_align = set()
        self.ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        if self.args.is_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.model = BootEA(kgs, args)
        self.model.init()
        self.model.to(self.device)
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        self.training_batch_queue = manager.Queue()
        self.optimizer = get_optimizer_torch(self.args.optimizer, self.model, self.args.learning_rate)

    def launch_training_k_epo(self, iter, iter_nums, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                              neighbors2):
        for i in range(1, iter_nums + 1):
            epoch = (iter - 1) * iter_nums + i
            self.launch_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                                             neighbors2)

    def generate_batch(self, steps_tasks, triple_steps):
        manager = mp.Manager()
        batch_queue = manager.Queue()
        self.training_batch_queue = []
        for steps_task in steps_tasks:
            mp.Process(target=batch.generate_relation_triple_batch_queue,
                       args=(self.kgs.kg1.relation_triples_list, self.kgs.kg2.relation_triples_list,
                             self.kgs.kg1.relation_triples_set, self.kgs.kg2.relation_triples_set,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, None, None, self.args.neg_triple_num)).start()
        # self.training_batch_queue = batch_queue
        for i in range(triple_steps):
            self.training_batch_queue.append(batch_queue.get())

    def launch_triple_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=batch.generate_relation_triple_batch_queue,
                       args=(self.kgs.kg1.relation_triples_list, self.kgs.kg2.relation_triples_list,
                             self.kgs.kg1.relation_triples_set, self.kgs.kg2.relation_triples_set,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2, self.args.neg_triple_num)).start()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            self.optimizer.zero_grad()
            batch_pos, batch_neg = batch_queue.get()
            trained_samples_num += len(batch_pos)
            batch_loss = self.model.define_embed_graph(
                {'pos_hs': to_tensor([x[0] for x in batch_pos], self.device),
                 'pos_rs': to_tensor([x[1] for x in batch_pos], self.device),
                 'pos_ts': to_tensor([x[2] for x in batch_pos], self.device),
                 'neg_hs': to_tensor([x[0] for x in batch_neg], self.device),
                 'neg_rs': to_tensor([x[1] for x in batch_neg], self.device),
                 'neg_ts': to_tensor([x[2] for x in batch_neg], self.device)})
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def train_alignment(self, kg1: KG, kg2: KG, entities1, entities2, training_epochs):
        if entities1 is None or len(entities1) == 0:
            return
        newly_tris1, newly_tris2 = generate_supervised_triples(kg1.rt_dict, kg1.hr_dict, kg2.rt_dict, kg2.hr_dict,
                                                               entities1, entities2)
        steps = math.ceil(((len(newly_tris1) + len(newly_tris2)) / self.args.batch_size))
        if steps == 0:
            steps = 1
        for i in range(training_epochs):
            t1 = time.time()
            alignment_loss = 0
            for step in range(steps):
                newly_batch1, newly_batch2 = generate_pos_batch(newly_tris1, newly_tris2, step, self.args.batch_size)
                newly_batch1.extend(newly_batch2)
                self.optimizer.zero_grad()
                alignment_feed_dict = {'new_h': to_tensor([tr[0] for tr in newly_batch1], self.device),
                                       'new_r': to_tensor([tr[1] for tr in newly_batch1], self.device),
                                       'new_t': to_tensor([tr[2] for tr in newly_batch1], self.device)}
                loss = self.model.define_alignment_graph(alignment_feed_dict)
                loss.backward()
                self.optimizer.step()
                alignment_loss += loss
            alignment_loss /= (len(newly_tris1) + len(newly_tris2))
            print("alignment_loss = {:.3f}, time = {:.3f} s".format(alignment_loss, time.time() - t1))

    def likelihood(self, labeled_alignment):
        t = time.time()
        likelihood_mat = calculate_likelihood_mat(self.ref_ent1, self.ref_ent2, labeled_alignment)
        likelihood_loss = 0.0
        steps = len(self.ref_ent1) // self.args.likelihood_slice
        ref_ent1_array = np.array(self.ref_ent1)
        ll = list(range(len(self.ref_ent1)))
        # print(steps)
        for i in range(steps):
            idx = random.sample(ll, self.args.likelihood_slice)
            self.optimizer.zero_grad()
            likelihood_feed_dict = {'entities1': to_tensor(ref_ent1_array[idx], self.device),
                                    'entities2': to_tensor(self.ref_ent2, self.device),
                                    'likelihood_mat': to_tensor(likelihood_mat[idx, :], self.device)}
            vals = self.model.define_likelihood_graph(likelihood_feed_dict)
            vals.backward()
            self.optimizer.step()
            likelihood_loss += vals
        print("likelihood_loss = {:.3f}, time = {:.3f} s".format(likelihood_loss, time.time() - t))

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        neighbors1, neighbors2 = None, None
        labeled_align = set()
        sub_num = self.args.sub_epoch
        iter_nums = self.args.max_epoch // sub_num
        for i in range(1, iter_nums + 1):
            print("\niteration", i)
            self.launch_training_k_epo(i, sub_num, triple_steps, steps_tasks, self.training_batch_queue, neighbors1,
                                       neighbors2)
            if i * sub_num >= self.args.start_valid:
                flag = self.model.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.args.no_early:
                    self.early_stop = False
                if self.early_stop or i == iter_nums:
                    break
            labeled_align, entities1, entities2 = bootstrapping(self.model.eval_ref_sim_mat(),
                                                                self.ref_ent1, self.ref_ent2, labeled_align,
                                                                self.args.sim_th, self.args.k)
            self.train_alignment(self.kgs.kg1, self.kgs.kg2, entities1, entities2, 1)
            # self.likelihood(labeled_align)
            if i * sub_num >= self.args.start_valid:
                self.model.valid(self.args.stop_metric)
            t1 = time.time()
            assert 0.0 < self.args.truncated_epsilon < 1.0
            neighbors_num1 = int((1 - self.args.truncated_epsilon) * self.kgs.kg1.entities_num)
            neighbors_num2 = int((1 - self.args.truncated_epsilon) * self.kgs.kg2.entities_num)
            if neighbors1 is not None:
                del neighbors1, neighbors2
            gc.collect()
            # neighbors1 = bat.generate_neighbours(self.eval_kg1_useful_ent_embeddings(),
            #                                      self.kgs.useful_entities_list1,
            #                                      neighbors_num1, self.args.batch_threads_num)
            # neighbors2 = bat.generate_neighbours(self.eval_kg2_useful_ent_embeddings(),
            #                                      self.kgs.useful_entities_list2,
            #                                      neighbors_num2, self.args.batch_threads_num)
            neighbors1 = batch.generate_neighbours_single_thread(self.model.eval_kg1_useful_ent_embeddings(),
                                                                 self.kgs.useful_entities_list1,
                                                                 neighbors_num1, self.args.test_threads_num)
            neighbors2 = batch.generate_neighbours_single_thread(self.model.eval_kg2_useful_ent_embeddings(),
                                                                 self.kgs.useful_entities_list2,
                                                                 neighbors_num2, self.args.test_threads_num)
            ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
            print("generating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
        self.test()
        self.model.save()

    def train_triple(self, i):
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / (self.args.batch_size * self.args.parallel_num)))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        sub_num = self.args.sub_epoch
        print("\niteration", i)
        self.launch_training_k_epo(i, sub_num, triple_steps, steps_tasks, self.training_batch_queue, None,
                                   None)
        gc.collect()
        # print("generating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))

    def train_mapping(self):
        self.labeled_align, entities1, entities2 = bootstrapping(self.model.eval_ref_sim_mat(),
                                                            self.ref_ent1, self.ref_ent2, self.labeled_align,
                                                            self.args.sim_th, self.args.k)
        self.train_alignment(self.kgs.kg1, self.kgs.kg2, entities1, entities2, 1)

    def test(self):
        rest_12 = self.model.test(self.kgs.test_entities1, self.kgs.test_entities2)

    def valid(self):
        return self.model.valid(self.args.stop_metric)

    def set_triple_list(self, triple_list_kg1, triple_list_kg2):
        self.kgs.kg1.relation_triples_list = triple_list_kg1
        self.kgs.kg1.relation_triples_set = set(triple_list_kg1)
        self.kgs.kg2.relation_triples_list = triple_list_kg2
        self.kgs.kg2.relation_triples_set = set(triple_list_kg2)


