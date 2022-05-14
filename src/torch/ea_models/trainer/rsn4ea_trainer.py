import math
import multiprocessing as mp
import random
import time
import os
import itertools
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from sklearn import preprocessing
from torch.autograd import Variable
import pandas as pd
from scipy.sparse import csr_matrix

from src.py.base.optimizers import get_optimizer_torch
from src.py.util.util import to_tensor, early_stop
from src.torch.ea_models.models.rsn4ea import RSN4EA
from src.torch.kge_models.basic_model import align_model_trainer


class rsn4ea_trainer(align_model_trainer):
    def __init__(self):
        super(rsn4ea_trainer, self).__init__()
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = None
        self.criterion = None
        self._train_data = None
        self._options = None
        self._r_m = None
        self._e_m = None
        self._kb = None
        self._rel_testing = None
        self._ent_testing = None
        self._rel_mapping = None
        self._ent_mapping = None
        self.optimizer = None
        self._rel_num = None
        self._ent_num = None

    def read(self, data_path='data/dbp_wd_15k_V1/mapping/0_3/'):
        # add shortcuts
        kgs = self.kgs

        kg1 = pd.DataFrame(kgs.kg1.relation_triples_list, columns=['h_id', 'r_id', 't_id'])
        kg2 = pd.DataFrame(kgs.kg2.relation_triples_list, columns=['h_id', 'r_id', 't_id'])

        kb = pd.concat([kg1, kg2], ignore_index=True)

        # self._eid_1 = pd.Series(eid_1)
        # self._eid_2 = pd.Series(eid_2)

        self._ent_num = kgs.entities_num
        self._rel_num = kgs.relations_num
        # self._ent_id = e_map
        # self._rel_id = r_map
        self._ent_mapping = pd.DataFrame(list(kgs.train_links), columns=['kb_1', 'kb_2'])
        self._rel_mapping = pd.DataFrame({}, columns=['kb_1', 'kb_2'])
        self._ent_testing = pd.DataFrame(list(kgs.test_links), columns=['kb_1', 'kb_2'])
        self._rel_testing = pd.DataFrame({}, columns=['kb_1', 'kb_2'])

        # add reverse edges
        rev_kb = kb[['t_id', 'r_id', 'h_id']].values
        rev_kb[:, 1] += self._rel_num
        rev_kb = pd.DataFrame(rev_kb, columns=['h_id', 'r_id', 't_id'])
        self._rel_num *= 2
        kb = pd.concat([kb, rev_kb], ignore_index=True)
        # print(kb)
        # print(kb[len(kb)//2:])

        self._kb = kb
        # we first tag the entities that have algined entities according to entity_mapping
        self.add_align_infor()
        # we then connect two KGs by creating new triples involving aligned entities.
        self.add_weight()

    def add_align_infor(self):
        kb = self._kb

        ent_mapping = self._ent_mapping
        rev_e_m = ent_mapping.rename(columns={'kb_1': 'kb_2', 'kb_2': 'kb_1'})
        rel_mapping = self._rel_mapping
        rev_r_m = rel_mapping.rename(columns={'kb_1': 'kb_2', 'kb_2': 'kb_1'})

        ent_mapping = pd.concat([ent_mapping, rev_e_m], ignore_index=True)
        rel_mapping = pd.concat([rel_mapping, rev_r_m], ignore_index=True)

        ent_mapping = pd.Series(ent_mapping.kb_2.values, index=ent_mapping.kb_1.values)
        rel_mapping = pd.Series(rel_mapping.kb_2.values, index=rel_mapping.kb_1.values)

        self._e_m = ent_mapping
        self._r_m = rel_mapping

        kb['ah_id'] = kb.h_id
        kb['ar_id'] = kb.r_id
        kb['at_id'] = kb.t_id

        h_mask = kb.h_id.isin(ent_mapping)
        r_mask = kb.r_id.isin(rel_mapping)
        t_mask = kb.t_id.isin(ent_mapping)

        kb['ah_id'][h_mask] = ent_mapping.loc[kb['ah_id'][h_mask].values]
        kb['ar_id'][r_mask] = rel_mapping.loc[kb['ar_id'][r_mask].values]
        kb['at_id'][t_mask] = ent_mapping.loc[kb['at_id'][t_mask].values]

        self._kb = kb

    def add_weight(self):
        kb = self._kb[['h_id', 'r_id', 't_id', 'ah_id', 'ar_id', 'at_id']]

        kb['w_h'] = 0
        kb['w_r'] = 0
        kb['w_t'] = 0

        h_mask = ~(kb.h_id == kb.ah_id)
        r_mask = ~(kb.r_id == kb.ar_id)
        t_mask = ~(kb.t_id == kb.at_id)

        kb.loc[h_mask, 'w_h'] = 1
        kb.loc[r_mask, 'w_r'] = 1
        kb.loc[t_mask, 'w_t'] = 1

        akb = kb[['ah_id', 'ar_id', 'at_id', 'w_h', 'w_r', 'w_t']]
        akb = akb.rename(columns={'ah_id': 'h_id', 'ar_id': 'r_id', 'at_id': 't_id'})

        ahkb = kb[h_mask][['ah_id', 'r_id', 't_id', 'w_h', 'w_r', 'w_t']].rename(columns={'ah_id': 'h_id'})
        arkb = kb[r_mask][['h_id', 'ar_id', 't_id', 'w_h', 'w_r', 'w_t']].rename(columns={'ar_id': 'r_id'})
        atkb = kb[t_mask][['h_id', 'r_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(columns={'at_id': 't_id'})
        ahrkb = kb[h_mask & r_mask][['ah_id', 'ar_id', 't_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ah_id': 'h_id', 'ar_id': 'r_id'})
        ahtkb = kb[h_mask & t_mask][['ah_id', 'r_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ah_id': 'h_id', 'at_id': 't_id'})
        artkb = kb[r_mask & t_mask][['h_id', 'ar_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ar_id': 'r_id', 'at_id': 't_id'})
        ahrtkb = kb[h_mask & r_mask & t_mask][['ah_id', 'ar_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ah_id': 'h_id',
                     'ar_id': 'r_id',
                     'at_id': 't_id'})

        kb['w_h'] = 0
        kb['w_r'] = 0
        kb['w_t'] = 0

        kb = pd.concat(
            [akb, ahkb, arkb, atkb, ahrkb, ahtkb, artkb, ahrtkb, kb[['h_id', 'r_id', 't_id', 'w_h', 'w_r', 'w_t']]],
            ignore_index=True).drop_duplicates()

        self._kb = kb.reset_index(drop=True)

    def sample_paths(self, repeat_times=2):
        opts = self._options

        kb = self._kb.copy()

        kb = kb[['h_id', 'r_id', 't_id']]

        # sampling triples with the h_id-(r_id,t_id) form.

        rtlist = np.unique(kb[['r_id', 't_id']].values, axis=0)

        rtdf = pd.DataFrame(rtlist, columns=['r_id', 't_id'])

        rtdf = rtdf.reset_index().rename({'index': 'tail_id'}, axis='columns')

        rtkb = kb.merge(
            rtdf, left_on=['r_id', 't_id'], right_on=['r_id', 't_id'])

        htail = np.unique(rtkb[['h_id', 'tail_id']].values, axis=0)

        htailmat = csr_matrix((np.ones(len(htail)), (htail[:, 0], htail[:, 1])),
                              shape=(self._ent_num, rtlist.shape[0]))

        # calulate corss-KG bias at first
        em = pd.concat(
            [self._ent_mapping.kb_1, self._ent_mapping.kb_2]).values

        rtkb['across'] = rtkb.t_id.isin(em)
        rtkb.loc[rtkb.across, 'across'] = opts.beta
        rtkb.loc[rtkb.across == 0, 'across'] = 1 - opts.beta

        rtailkb = rtkb[['h_id', 't_id', 'tail_id', 'across']]

        def gen_tail_dict(x):
            return x.tail_id.values, x.across.values / x.across.sum()

        rtailkb = rtailkb.groupby('h_id').apply(gen_tail_dict)

        rtailkb = pd.DataFrame({'tails': rtailkb})

        # start sampling

        hrt = np.repeat(kb.values, repeat_times, axis=0)

        # for starting triples
        def perform_random(x):
            return np.random.choice(x.tails[0], 1, p=x.tails[1].astype(np.float))

        # else
        def perform_random2(x):
            # calculate depth bias
            pre_c = htailmat[np.repeat(x.pre, x.tails[0].shape[0]), x.tails[0]]
            pre_c[pre_c == 0] = opts.alpha
            pre_c[pre_c == 1] = 1 - opts.alpha
            p = x.tails[1].astype(np.float).reshape(
                [-1, ]) * pre_c.A.reshape([-1, ])
            p = p / p.sum()
            return np.random.choice(x.tails[0], 1, p=p)

        # print(rtailkb.loc[hrt[:, 2]])
        rt_x = rtailkb.loc[hrt[:, 2]].apply(perform_random, axis=1)
        rt_x = rtlist[np.concatenate(rt_x.values)]

        rts = [hrt, rt_x]
        print('hrt', 'rt_x', len(hrt), len(rt_x))
        c_length = 5
        while c_length < opts.max_length:
            curr = rtailkb.loc[rt_x[:, 1]]
            print(len(curr), len(hrt[:, 0]))
            curr.loc[:, 'pre'] = hrt[:, 0]

            rt_x = curr.apply(perform_random2, axis=1)
            rt_x = rtlist[np.concatenate(rt_x.values)]

            rts.append(rt_x)
            c_length += 2

        data = np.concatenate(rts, axis=1)
        data = pd.DataFrame(data)
        self._train_data = data
        print("save paths to:", '%spaths_%.1f_%.1f' % (opts.data_path, opts.alpha, opts.beta))
        data.to_csv('%spaths_%.1f_%.1f' % (opts.data_path, opts.alpha, opts.beta))

    def init(self, args, kgs):
        self.args = args
        self.kgs = kgs
        if self.args.is_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self._options = opts = self.args
        opts.data_path = opts.training_data

        self.read(data_path=self._options.data_path)

        sequence_datapath = '%spaths_%.1f_%.1f' % (
            self._options.data_path, self._options.alpha, self._options.beta)

        if not os.path.exists(sequence_datapath):
            self.sample_paths()
        else:
            print('load existing training sequences')
            self._train_data = pd.read_csv('%spaths_%.1f_%.1f' % (
                self._options.data_path, self._options.alpha, self._options.beta), index_col=0)
        self.model = RSN4EA(kgs, args)
        self.model.initial(self._ent_num, self._rel_num)
        self.model.to(self.device)

    # training procedure
    def seq_train(self, data, choices=None, epoch=None):
        opts = self._options

        choices = np.random.choice(len(data), size=len(data), replace=True)
        batch_size = opts.batch_size

        num_batch = len(data) // batch_size
        losses = 0
        for i in range(num_batch):
            one_batch_choices = choices[i * batch_size: (i + 1) * batch_size]
            one_batch_data = data.iloc[one_batch_choices]
            self.optimizer.zero_grad()
            seq = one_batch_data.values[:, :opts.max_length]
            loss = self.model.build_sub_graph(opts.max_length, False, {"seq":to_tensor(seq, self.device)})
            loss.backward()
            self.optimizer.step()
            del one_batch_data
            losses += loss
        last_mean_loss = losses / num_batch

        return last_mean_loss

    def test(self):
        rest_12 = self.model.tests(self.kgs.test_entities1, self.kgs.test_entities2)

    def run(self):
        t = time.time()
        self.optimizer = get_optimizer_torch(self.args.optimizer, self.model, self.args.learning_rate)
        train_data = self._train_data
        for i in range(1, self.args.max_epoch + 1):
            time_i = time.time()
            last_mean_loss = self.seq_train(train_data)
            print('epoch %i, avg. batch_loss: %f,  cost time: %.4f s' % (i, last_mean_loss, time.time() - time_i))
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.model.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.args.no_early:
                    self.early_stop = False
                if self.early_stop or i == self.args.max_epoch:
                    break
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
