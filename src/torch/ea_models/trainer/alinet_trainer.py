import gc
import math
import os
import pickle
import random
import time
import torch
import torch.nn
import numpy as np
from torch.autograd import Variable

from src.py.base.optimizers import get_optimizer_torch
from src.py.evaluation.evaluation import valid, test
from src.py.load import read
from src.py.util.util import to_var, generate_out_folder, to_tensor_cpu, to_tensor
from src.torch.ea_models.models.alinet import AKG, enhance_triples, remove_unlinked_triples, generate_rel_ht, \
    no_weighted_adj, generate_2hop_triples, generate_neighbours, AliNet
from src.torch.ea_models.models.gcn_align import GCN_Utils, load_attr, GCN_Align_Unit
from src.torch.kge_models.basic_model import align_model_trainer


def early_stop(flag1, flag2, flag):
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False


class alinet_trainer():
    def __init__(self):
        super(alinet_trainer, self).__init__()
        self.sup_links_set = None
        self.new_sup_links_set = None
        self.neg_link_batch = None
        self.pos_link_batch = None
        self.sup_links = None
        self.sim_th = None
        self.rel_win_size = None
        self.rel_ht_dict = None
        self.linked_ents = None
        self.sup_ent2 = None
        self.sup_ent1 = None
        self.ref_ent2 = None
        self.ref_ent1 = None
        self.is_two = None
        self.out_folder = None
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = None
        self.optimizer = None
        self.attr = None
        self.opt = 'Adam'
        self.act_func = torch.relu
        self.dropout = 0.0
        self.struct_loss = None
        self.embed_test1 = None
        self.embed_test2 = None
        self.struct_optimizer = None
        self.vec_ae = None
        self.vec_se = None
        self.num_supports = None
        self.utils = None
        self.adj = None
        self.ae_input = None
        self.train = None
        self.e = None
        self.support = None
        self.adj = None
        self.ph_ae = None
        self.ph_se = None
        self.model = None

    def init(self, args, kgs):
        self.args = args
        self.kgs = kgs
        if self.args.is_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.model = AliNet()
        self.model.set_kgs(kgs)
        self.model.set_args(args)
        self.model.init()
        self.model.to(self.device)
        self.optimizer = get_optimizer_torch(self.args.optimizer, self.model, self.args.learning_rate)
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division,
                                              'Alinet')

    def test(self, save=True):
        if self.embed_test1 is None:
            embeds1, embeds2, _ = self.model._eval_test_embeddings()
        else:
            embeds1 = self.embed_test1
            embeds2 = self.embed_test2
        rest_12, _, _, _ = test(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
                                metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        test(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls, accurate=True)
        '''if save:
            ent_ids_rest_12 = [(self.kgs.test_entities1[i], self.kgs.test_entities2[j]) for i, j in rest_12]
            rd.save_results(self.out_folder, ent_ids_rest_12)'''

    def retest(self, save=True):
        if self.embed_test1 is None:
            embeds1, embeds2, _ = self.model._eval_test_embeddings()
        else:
            embeds1 = self.embed_test1
            embeds2 = self.embed_test2
        rest_12, _, _, _ = test(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
                                metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        test(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls, accurate=True)

    def save(self):
        embeds1, embeds2, _ = self.model._eval_test_embeddings()
        read.save_embeddings(self.out_folder, self.kgs, embeds1, embeds2, None, mapping_mat=None)

    def load(self):
        dir = self.out_folder.split("/")
        new_dir = ""
        print(dir)
        for i in range(len(dir) - 1):
            new_dir += (dir[i] + "/")
        exist_file = os.listdir(new_dir)
        new_dir = new_dir + "/"
        embeds1 = np.load(new_dir + "ent_embeds.npy")
        embeds2 = np.load(new_dir + "rel_embeds.npy")
        self.embed_test1 = embeds1
        self.embed_test2 = embeds2

    def valid_(self, stop_metric):
        embeds1, embeds2, _ = self.model._eval_valid_embeddings()
        hits1_12, mrr_12 = valid(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
                                 metric=self.args.eval_metric)
        if stop_metric == 'hits1':
            return hits1_12
        return mrr_12

    def generate_input_batch(self, batch_size, neighbors1=None, neighbors2=None):
        if batch_size > len(self.model.sup_ent1):
            batch_size = len(self.model.sup_ent1)
        index = np.random.choice(len(self.model.sup_ent1), batch_size)
        pos_links = self.model.sup_links[index,]
        neg_links = list()
        if neighbors1 is None:
            neg_ent1 = list()
            neg_ent2 = list()
            for i in range(self.args.neg_triple_num):
                neg_ent1.extend(random.sample(self.model.sup_ent1 + self.model.ref_ent1, batch_size))
                neg_ent2.extend(random.sample(self.model.sup_ent2 + self.model.ref_ent2, batch_size))
            neg_links.extend([(neg_ent1[i], neg_ent2[i]) for i in range(len(neg_ent1))])
        else:
            for i in range(batch_size):
                e1 = pos_links[i, 0]
                candidates = random.sample(neighbors1.get(e1), self.args.neg_triple_num)
                neg_links.extend([(e1, candidate) for candidate in candidates])
                e2 = pos_links[i, 1]
                candidates = random.sample(neighbors2.get(e2), self.args.neg_triple_num)
                neg_links.extend([(candidate, e2) for candidate in candidates])
        neg_links = set(neg_links) - self.model.sup_links_set
        neg_links = neg_links - self.model.new_sup_links_set
        neg_links = np.array(list(neg_links))
        return pos_links, neg_links

    def generate_rel_batch(self):
        hs, rs, ts = list(), list(), list()
        for r, hts in self.model.rel_ht_dict.items():
            hts_batch = [random.choice(hts) for _ in range(self.model.rel_win_size)]
            for h, t in hts_batch:
                hs.append(h)
                ts.append(t)
                rs.append(r)
        return hs, rs, ts

    def run(self):
        flag1 = 0
        flag2 = 0
        steps = len(self.model.sup_ent2) // self.args.batch_size
        neighbors1, neighbors2 = None, None
        if steps == 0:
            steps = 1
        for epoch in range(1, self.args.max_epoch + 1):
            start = time.time()
            epoch_loss = 0.0
            for step in range(steps):
                self.pos_link_batch, self.neg_link_batch = self.generate_input_batch(self.args.batch_size,
                                                                                     neighbors1=neighbors1,
                                                                                     neighbors2=neighbors2)
                self.optimizer.zero_grad()
                self.model._define_model()
                if self.args.rel_param > 0:
                    hs, _, ts = self.generate_rel_batch()
                    loss = self.model.compute_loss(to_tensor(self.pos_link_batch, self.device),
                                                   to_tensor(self.neg_link_batch,
                                                             self.device)) + self.model.compute_rel_loss(
                        to_tensor(hs, self.device), to_tensor(ts, self.device))
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                else:
                    loss = self.model.compute_loss(to_tensor(self.pos_link_batch, self.device),
                                                   to_tensor(self.neg_link_batch, self.device))
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
            print('epoch {}, loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))
            if epoch % self.args.eval_freq == 0 and epoch >= self.args.start_valid:
                flag = self.valid_(self.args.stop_metric)
                flag1, flag2, is_stop = early_stop(flag1, flag2, flag)
                if self.args.no_early:
                    is_stop = False
                if is_stop:
                    print("\n == training stop == \n")
                    break
                neighbors1, neighbors2 = self.model.find_neighbors()
                if epoch >= self.args.start_augment * self.args.eval_freq:
                    if self.args.sim_th > 0.0:
                        self.model.augment_neighborhood()
        self.save()
