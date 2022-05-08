#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.py.load.kg import parse_triples
from src.py.util.util import to_tensor_cpu


class PyTorchTrainDataset(Dataset):

    def __init__(self, triples, neg_num, kgs):
        # triples
        self.head = [x[0] for x in triples]
        self.tail = [x[2] for x in triples]
        self.rel = [x[1] for x in triples]
        # total numbers of entities, relations, and triples
        self.neg_num = neg_num
        self.kgs = kgs

    def __len__(self):
        return len(self.head)

    def __getitem__(self, idx):
        return self.head[idx], self.rel[idx], self.tail[idx]

    def collate_fn(self, data):
        batch_h = [item[0] for item in data]
        batch_r = [item[1] for item in data]
        batch_t = [item[2] for item in data]
        batch_neg = self.generate_neg_triples_fast(data, set(self.kgs.relation_triples_list), self.kgs.entities_list, self.neg_num)
        batch_data = list()
        batch_h = to_tensor_cpu(batch_h + [x[0] for x in batch_neg])
        batch_r = to_tensor_cpu(batch_r + [x[1] for x in batch_neg])
        batch_t = to_tensor_cpu(batch_t + [x[2] for x in batch_neg])
        batch_data.append(batch_h)
        batch_data.append(batch_r)
        batch_data.append(batch_t)
        batch_data = torch.stack(batch_data)
        """
        batch_data['batch_h'] = batch_h.squeeze()
        batch_data['batch_t'] = batch_t.squeeze()
        batch_data['batch_r'] = batch_r.squeeze()
        batch_data['batch_y'] = batch_y.squeeze()
        """
        return batch_data

    def generate_neg_triples_fast(self, pos_batch, all_triples_set, entities_list, neg_triples_num, neighbor=None,
                                  max_try=10):
        if neighbor is None:
            neighbor = dict()
        neg_batch = list()
        for head, relation, tail in pos_batch:
            neg_triples = list()
            nums_to_sample = neg_triples_num
            head_candidates = neighbor.get(head, entities_list)
            tail_candidates = neighbor.get(tail, entities_list)
            for i in range(max_try):
                corrupt_head_prob = np.random.binomial(1, 0.5)
                if corrupt_head_prob:
                    neg_heads = random.sample(head_candidates, nums_to_sample)
                    i_neg_triples = {(h2, relation, tail) for h2 in neg_heads}
                else:
                    neg_tails = random.sample(tail_candidates, nums_to_sample)
                    i_neg_triples = {(head, relation, t2) for t2 in neg_tails}
                if i == max_try - 1:
                    neg_triples += list(i_neg_triples)
                    break
                else:
                    i_neg_triples = list(i_neg_triples - all_triples_set)
                    neg_triples += i_neg_triples
                if len(neg_triples) == neg_triples_num:
                    break
                else:
                    nums_to_sample = neg_triples_num - len(neg_triples)
            assert len(neg_triples) == neg_triples_num
            neg_batch.extend(neg_triples)
        assert len(neg_batch) == neg_triples_num * len(pos_batch)
        return neg_batch

    def set_sampling_mode(self, sampling_mode):
        self.sampling_mode = sampling_mode

    def set_ent_neg_rate(self, rate):
        self.neg_ent = rate

    def set_rel_neg_rate(self, rate):
        self.neg_rel = rate

    def set_bern_flag(self, bern_flag):
        self.bern_flag = bern_flag

    def set_filter_flag(self, filter_flag):
        self.filter_flag = filter_flag

    def get_ent_tot(self):
        return self.ent_total

    def get_rel_tot(self):
        return self.rel_total

    def get_tri_tot(self):
        return self.tri_total


class PyTorchTrainDataLoader(DataLoader):
    def __init__(self, kgs, batch_size, threads, neg_size):
        self.batch_size = batch_size
        self.kgs = kgs
        self.neg_size = neg_size
        self.data = self.__construct_dataset()
        super(PyTorchTrainDataLoader, self).__init__(
            dataset=self.data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=threads,
            pin_memory=True,
            collate_fn=self.data.collate_fn,
            drop_last=False
        )

    def __construct_dataset(self):
        triples_set = self.kgs.relation_triples_set
        train_dataset = PyTorchTrainDataset(list(triples_set), self.kgs.entities_num,
                                            self.kgs.relations_num, neg_ent=self.neg_size)
        return train_dataset

    def get_ent_tot(self):
        return self.data.get_ent_tot()

    def get_rel_tot(self):
        return self.data.get_rel_tot()

    def get_batch_size(self):
        return self.batch_size

    """interfaces to set essential parameters"""

    def set_sampling_mode(self, sampling_mode):
        self.dataset.set_sampling_mode(sampling_mode)

    def set_work_threads(self, work_threads):
        self.num_workers = work_threads

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches
        self.batch_size = self.tripleTotal // self.nbatches

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.nbatches = self.tripleTotal // self.batch_size

    def set_ent_neg_rate(self, rate):
        self.dataset.set_ent_neg_rate(rate)

    def set_rel_neg_rate(self, rate):
        self.dataset.set_rel_neg_rate(rate)

    def set_bern_flag(self, bern_flag):
        self.dataset.set_bern_flag(bern_flag)

    def set_filter_flag(self, filter_flag):
        self.dataset.set_filter_flag(filter_flag)

    """interfaces to get essential parameters"""

    def get_batch_size(self):
        return self.batch_size

    def get_ent_tot(self):
        return self.dataset.get_ent_tot()

    def get_rel_tot(self):
        return self.dataset.get_rel_tot()

    def get_triple_tot(self):
        return self.dataset.get_tri_tot()


def parse_triples_list(relation_set):
    subjects, predicates, objects = list(), list(), list()
    for o, p, s in relation_set:
        objects.append(o)
        predicates.append(p)
        subjects.append(s)
    return objects, predicates, subjects

"""
class PyTorchTrainKE(Dataset):

    def __init__(self, args, kgs):
        # triples
        self.args = args
        self.kgs = kgs
        self.head, self.rel, self.tail = parse_triples_list(self.kgs.relation_triples_set)
        # total numbers of entities, relations, and triples
        self.rel_total = self.kgs.relations_num
        self.ent_total = self.kgs.entities_num
        self.tri_total = len(self.head)
        # the sampling mode
        # the number of negative examples
        self.neg_ent = self.args.neg_triple_num
        self.neg_rel = 0
        self.bern_flag = True
        self.cross_sampling_flag = None
        self.filter_flag = True
        self.__count_htr()

    def __len__(self):
        return self.tri_total

    def __getitem__(self, idx):
        return self.head[idx], self.tail[idx], self.rel[idx]

    def collate_fn(self, data):
        batch_data = {}
        h = np.array([item[0] for item in data])
        t = np.array([item[1] for item in data])
        r = np.array([item[2] for item in data])
        batch_h = np.repeat(h.reshape(-1, 1), self.neg_ent + self.neg_rel, axis=-1)
        batch_t = np.repeat(t.reshape(-1, 1), self.neg_ent + self.neg_rel, axis=-1)
        batch_r = np.repeat(r.reshape(-1, 1), self.neg_ent + self.neg_rel, axis=-1)
        for index, item in enumerate(data):
            last = 0
            if self.neg_ent > 0:
                neg_head, neg_tail = self.__normal_batch(item[0], item[1], item[2], self.neg_ent)
                if len(neg_head) > 0:
                    batch_h[index][last:last + len(neg_head)] = neg_head
                    last += len(neg_head)
                if len(neg_tail) > 0:
                    batch_t[index][last:last + len(neg_tail)] = neg_tail
                    last += len(neg_tail)
        batch_h = np.concatenate((h, batch_h.reshape(-1)), 0)
        batch_r = np.concatenate((r, batch_r.reshape(-1)), 0)
        batch_t = np.concatenate((t, batch_t.reshape(-1)), 0)
        batch_h = to_tensor_cpu(batch_h)
        batch_t = to_tensor_cpu(batch_t)
        batch_r = to_tensor_cpu(batch_r)

        batch_data = []
        batch_data.append(batch_h.squeeze())
        batch_data.append(batch_r.squeeze())
        batch_data.append(batch_t.squeeze())
        return batch_data

    def __count_htr(self):

        self.h_of_rt = self.kgs.h_dict
        self.t_of_hr = self.kgs.t_dict
        self.r_of_ht = self.kgs.r_dict
        self.h_of_r = {}
        self.t_of_r = {}
        self.freqRel = {}
        self.lef_mean = {}
        self.rig_mean = {}

        triples = zip(self.head, self.rel, self.tail)
        for h, r, t in triples:
            if r not in self.freqRel:
                self.freqRel[r] = 0
                self.h_of_r[r] = {}
                self.t_of_r[r] = {}
            self.freqRel[r] += 1.0
            self.h_of_r[r][h] = 1
            self.t_of_r[r][t] = 1

        for r, t in self.h_of_rt:
            self.h_of_rt[(r, t)] = np.array(list(set(self.h_of_rt[(r, t)])))
        for h, r in self.t_of_hr:
            self.t_of_hr[(h, r)] = np.array(list(set(self.t_of_hr[(h, r)])))
        for h, t in self.r_of_ht:
            self.r_of_ht[(h, t)] = np.array(list(set(self.r_of_ht[(h, t)])))
        for r in range(self.rel_total):
            self.h_of_r[r] = np.array(list(self.h_of_r[r].keys()))
            self.t_of_r[r] = np.array(list(self.t_of_r[r].keys()))
            self.lef_mean[r] = self.freqRel[r] / len(self.h_of_r[r])
            self.rig_mean[r] = self.freqRel[r] / len(self.t_of_r[r])

    def __corrupt_head(self, t, r, num_max=1):
        tmp = torch.randint(low=0, high=self.ent_total, size=(num_max,)).numpy()
        if not self.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.h_of_rt[(r, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def __corrupt_tail(self, h, r, num_max=1):
        tmp = torch.randint(low=0, high=self.ent_total, size=(num_max,)).numpy()
        if not self.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.t_of_hr[(h, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def __corrupt_rel(self, h, t, num_max=1):
        tmp = torch.randint(low=0, high=self.rel_total, size=(num_max,)).numpy()
        if not self.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.r_of_ht[(h, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def __normal_batch(self, h, t, r, neg_size):
        neg_size_h = 0
        neg_size_t = 0
        prob = self.rig_mean[r] / (self.rig_mean[r] + self.lef_mean[r]) if self.bern_flag else 0.5
        for i in range(neg_size):
            if random.random() < prob:
                neg_size_h += 1
            else:
                neg_size_t += 1

        neg_list_h = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_h:
            neg_tmp_h = self.__corrupt_head(t, r, num_max=(neg_size_h - neg_cur_size) * 2)
            neg_list_h.append(neg_tmp_h)
            neg_cur_size += len(neg_tmp_h)
        if neg_list_h != []:
            neg_list_h = np.concatenate(neg_list_h)

        neg_list_t = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_t:
            neg_tmp_t = self.__corrupt_tail(h, r, num_max=(neg_size_t - neg_cur_size) * 2)
            neg_list_t.append(neg_tmp_t)
            neg_cur_size += len(neg_tmp_t)
        if neg_list_t != []:
            neg_list_t = np.concatenate(neg_list_t)

        return neg_list_h[:neg_size_h], neg_list_t[:neg_size_t]

    def __head_batch(self, h, t, r, neg_size):
        # return torch.randint(low = 0, high = self.ent_total, size = (neg_size, )).numpy()
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_head(t, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def __tail_batch(self, h, t, r, neg_size):
        # return torch.randint(low = 0, high = self.ent_total, size = (neg_size, )).numpy()
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_tail(h, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def __rel_batch(self, h, t, r, neg_size):
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_rel(h, t, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]
"""