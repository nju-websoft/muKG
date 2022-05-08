import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.py.evaluation.evaluation import valid, test
from src.py.load import read
from src.py.util.util import to_tensor
from src.torch.kge_models.basic_model import BasicModel


class SEA(BasicModel):

    def __init__(self, args, kgs):
        super(SEA, self).__init__(args, kgs)
        # self.out_folder = r'D:\OPENEA-pytorch\result'

    def init(self):
        self._define_variables()
        # customize parameters
        #assert self.args.init == 'unit'
        assert self.args.loss == 'margin-based'
        assert self.args.alignment_module == 'mapping'
        assert self.args.loss == 'margin-based'
        assert self.args.neg_sampling == 'uniform'
        # assert self.args.optimizer == 'Adam'
        assert self.args.eval_metric == 'inner'
        assert self.args.loss_norm == 'L2'
        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True
        # assert self.args.neg_triple_num == 1

    def _define_variables(self):
        self.ent_embeds = nn.Embedding(self.kgs.entities_num, self.args.dim)
        self.rel_embeds = nn.Embedding(self.kgs.relations_num, self.args.dim)

        self.mapping_matrix_1 = nn.Parameter(torch.Tensor(self.args.dim, self.args.dim), requires_grad=True)
        nn.init.orthogonal_(self.mapping_matrix_1)
        self.eye_mat_1 = torch.Tensor(self.args.dim, self.args.dim)
        nn.init.eye_(self.eye_mat_1)

        self.mapping_matrix_2 = nn.Parameter(torch.Tensor(self.args.dim, self.args.dim), requires_grad=True)
        nn.init.orthogonal_(self.mapping_matrix_2)
        self.eye_mat_2 = torch.Tensor(self.args.dim, self.args.dim)
        nn.init.eye_(self.eye_mat_2)

        self.margin = nn.Parameter(torch.Tensor([1.5]))
        self.margin.requires_grad = False
        if self.args.init == 'xavier':
            nn.init.xavier_uniform_(self.ent_embeds.weight.data)
            nn.init.xavier_uniform_(self.rel_embeds.weight.data)
        elif self.args.init == 'normal':
            std = 1.0 / math.sqrt(self.args.dim)
            nn.init.trunc_normal_(self.ent_embeds.weight.data, std=std)
            nn.init.trunc_normal_(self.rel_embeds.weight.data, std=std)
        elif self.args.init == 'uniform':
            nn.init.uniform_(self.ent_embeds.weight.data, 0)
            nn.init.uniform_(self.rel_embeds.weight.data, 0)
        #self.ent_embeds.weight.data = F.normalize(self.ent_embeds.weight.data, 2, -1)
        #self.rel_embeds.weight.data = F.normalize(self.rel_embeds.weight.data, 2, -1)

    def generate_transE_loss(self, data):
        ph = data['pos_hs']
        pr = data['pos_rs']
        pt = data['pos_ts']
        nh = data['neg_hs']
        nr = data['neg_rs']
        nt = data['neg_ts']
        batch_size_now = ph.shape[0]
        if self.args.ent_l2_norm:
          ph = F.normalize(self.ent_embeds(ph), 2, -1)
          pr = F.normalize(self.rel_embeds(pr), 2, -1)
          pt = F.normalize(self.ent_embeds(pt), 2, -1)
          nh = F.normalize(self.ent_embeds(nh), 2, -1)
          nr = F.normalize(self.rel_embeds(nr), 2, -1)
          nt = F.normalize(self.ent_embeds(nt), 2, -1)
        else:
          ph = self.ent_embeds(ph)
          pr = self.rel_embeds(pr)
          pt = self.ent_embeds(pt)
          nh = self.ent_embeds(nh)
          nr = self.rel_embeds(nr)
          nt = self.ent_embeds(nt)
        if self.args.loss_norm == "L2":
            pos = torch.pow(torch.norm(ph + pr - pt, 2, -1), 2)
            neg = torch.pow(torch.norm(nh + nr - nt, 2, -1), 2)
        else:
            pos = torch.norm(ph + pr - pt, 1, -1)
            neg = torch.norm(nh + nr - nt, 1, -1)
        pos = pos.view(batch_size_now, -1)
        neg = neg.view(batch_size_now, -1)
        return torch.sum(torch.relu_(pos - neg + self.margin))

    def generate_mapping_loss(self, data):
        seed_entity1_labeled = data['seed1_labeled']
        seed_entity2_labeled = data['seed2_labeled']
        seed_entity1_unlabeled = data['seed1_unlabeled']
        seed_entity2_unlabeled = data['seed2_unlabeled']
        if self.args.ent_l2_norm:
          seed_entity1_labeled = F.normalize(self.ent_embeds(seed_entity1_labeled), 2, -1)
          seed_entity2_labeled = F.normalize(self.ent_embeds(seed_entity2_labeled), 2, -1)
          seed_entity1_unlabeled = F.normalize(self.ent_embeds(seed_entity1_unlabeled), 2, -1)
          seed_entity2_unlabeled = F.normalize(self.ent_embeds(seed_entity2_unlabeled), 2, -1)
        else:
          seed_entity1_labeled = self.ent_embeds(seed_entity1_labeled)
          seed_entity2_labeled = self.ent_embeds(seed_entity2_labeled)
          seed_entity1_unlabeled = self.ent_embeds(seed_entity1_unlabeled)
          seed_entity2_unlabeled = self.ent_embeds(seed_entity2_unlabeled)
        map12 = F.normalize(torch.matmul(seed_entity1_labeled, self.mapping_matrix_1), 2, -1)
        map21 = F.normalize(torch.matmul(seed_entity2_labeled, self.mapping_matrix_2), 2, -1)
        map_loss12 = torch.sum(torch.pow(torch.norm(seed_entity2_labeled - map12, 2, -1), 2))
        map_loss21 = torch.sum(torch.pow(torch.norm(seed_entity1_labeled - map21, 2, -1), 2))
        semi_map121 = F.normalize(torch.matmul(torch.matmul(seed_entity1_unlabeled, self.mapping_matrix_1), self.mapping_matrix_2), 2, -1)
        semi_map212 = F.normalize(torch.matmul(torch.matmul(seed_entity2_unlabeled, self.mapping_matrix_2), self.mapping_matrix_1), 2, -1)
        map_loss11 = torch.sum(torch.pow(torch.norm(seed_entity1_unlabeled - semi_map121, 2, -1), 2))
        map_loss22 = torch.sum(torch.pow(torch.norm(seed_entity2_unlabeled - semi_map212, 2, -1), 2))
        mapping_loss = self.args.alpha_1 * (map_loss12 + map_loss21) + \
                       self.args.alpha_2 * (map_loss11 + map_loss22)
        return mapping_loss

    def valid(self, stop_metric):
        if len(self.kgs.valid_links) > 0:
            seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities1, self.device)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities2 + self.kgs.test_entities2, self.device)), 2, -1)
        else:
            seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities1, self.device)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities2, self.device)), 2, -1)
        hits1_12, mrr_12 = valid(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(),
                                 self.mapping_matrix_1.cpu().detach().numpy(),
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12

    def tests(self, entities1, entities2):
        seed_entity1 = F.normalize(self.ent_embeds(to_tensor(entities1, self.device)), 2, -1)
        seed_entity2 = F.normalize(self.ent_embeds(to_tensor(entities2, self.device)), 2, -1)
        _, _, _, sim_list = test(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(),
                                 self.mapping_matrix_1.cpu().detach().numpy(),
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric, normalize=self.args.eval_norm,
                                 csls_k=0, accurate=True)
        print()
        return sim_list

    def save(self):
        ent_embeds = self.ent_embeds.cpu().weight.data
        rel_embeds = self.rel_embeds.cpu().weight.data
        mapping_mat_1 = self.mapping_matrix_1.cpu().detach().numpy()
        mapping_mat_2 = self.mapping_matrix_2.cpu().detach().numpy()
        read.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None,
                           mapping_mat=mapping_mat_1, rev_mapping_mat=mapping_mat_2)
