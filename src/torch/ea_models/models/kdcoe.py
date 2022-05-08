import math
import random
import string
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.basic_model import BasicModel
from src.modules.data_loader import LoadDataset
from torch.autograd import Variable
import pandas as pd
from src.modules.finding.alignment import search_nearest_k
from src.modules.finding.evaluation import valid, test
from src.modules.finding.similarity import sim
from src.trainer.util import to_tensor


class KDCoE(BasicModel):
    def __init__(self, args, kgs):
        super().__init__(args, kgs)
        self.desc_embedding2 = None
        self.desc_embedding1 = None
        self.desc_batch_size = None
        self.negative_indication_weight = None
        self.wv_dim = None
        self.default_desc_length = None
        self.word_embed = None

        self.desc_sim_th = None
        self.sim_th = None

        self.word_em = None
        self.e_desc = None

        self.ref_entities1 = None
        self.ref_entities2 = None

        self.new_alignment = set()
        self.new_alignment_index = set()

    def init(self):

        assert self.args.alpha > 1

        self.desc_batch_size = self.args.desc_batch_size
        self.negative_indication_weight = -1. / self.desc_batch_size
        self.wv_dim = self.args.wv_dim
        self.default_desc_length = self.args.default_desc_length
        self.word_embed = self.args.word_embed
        self.desc_sim_th = self.args.desc_sim_th
        self.sim_th = self.args.sim_th

        # self.word_em, self.e_desc = self._get_desc_input()

        self.ref_entities1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_entities2 = self.kgs.valid_entities2 + self.kgs.test_entities2

        self._define_variables()

    def _define_variables(self):
        self.ent_embeds = nn.Embedding(self.kgs.entities_num, self.args.dim)
        self.rel_embeds = nn.Embedding(self.kgs.relations_num, self.args.dim)

        self.mapping_matrix = Variable(torch.Tensor(self.args.dim, self.args.dim), requires_grad=True)
        nn.init.orthogonal_(self.mapping_matrix, gain=1)
        self.eye_mat = torch.Tensor(self.args.dim, self.args.dim)
        nn.init.eye_(self.eye_mat)

        self.margin = nn.Parameter(torch.Tensor([1]))
        self.margin.requires_grad = False
        if self.args.init == 'xavier':
            nn.init.xavier_uniform_(self.ent_embeds.weight.data)
            nn.init.xavier_uniform_(self.rel_embeds.weight.data)
        elif self.args.init == 'normal':
            std = 1.0 / math.sqrt(self.args.dim)
            nn.init.normal_(self.ent_embeds.weight.data, 0, std)
            nn.init.normal_(self.rel_embeds.weight.data, 0, std)
        elif self.args.init == 'uniform':
            nn.init.uniform_(self.ent_embeds.weight.data, 0)
            nn.init.uniform_(self.rel_embeds.weight.data, 0)

    def generate_transE_loss(self, data):
        ph = data['pos_hs']
        pr = data['pos_rs']
        pt = data['pos_ts']
        nh = data['neg_hs']
        nr = data['neg_rs']
        nt = data['neg_ts']
        batch_size_now = ph.shape[0]
        ph = F.normalize(self.ent_embeds(ph), 2, -1)
        pr = F.normalize(self.rel_embeds(pr), 2, -1)
        pt = F.normalize(self.ent_embeds(pt), 2, -1)
        nh = F.normalize(self.ent_embeds(nh), 2, -1)
        nr = F.normalize(self.rel_embeds(nr), 2, -1)
        nt = F.normalize(self.ent_embeds(nt), 2, -1)
        if self.args.loss_norm == "L2":
            pos = torch.norm(ph + pr - pt, 2, -1)
            neg = torch.norm(nh + nr - nt, 2, -1)
        else:
            pos = torch.norm(ph + pr - pt, 1, -1)
            neg = torch.norm(nh + nr - nt, 1, -1)
        pos = pos.view(batch_size_now, -1)
        neg = neg.view(batch_size_now, -1)
        return torch.sum(torch.relu_(pos - neg + self.margin))

    def generate_mapping_loss(self, data):
        seed_entity1 = data['seed1']
        seed_entity2 = data['seed2']
        seed_entity1 = F.normalize(self.ent_embeds(seed_entity1), 2, -1)
        seed_entity2 = F.normalize(self.ent_embeds(seed_entity2), 2, -1)
        seed_mapped_entity = F.normalize(torch.matmul(seed_entity1, self.mapping_matrix), 2, -1)
        distance = seed_mapped_entity - seed_entity2
        align_loss = torch.sum(torch.norm(distance, 2, -1))
        orthogonal_loss = torch.mean(
            torch.sum(torch.pow(torch.matmul(self.mapping_matrix, self.mapping_matrix.t()) - self.eye_mat, 2), -1)
        )
        return align_loss + orthogonal_loss

    def define_desc_graph(self, data):
        desc1 = data['desc1']
        desc2 = data['desc2']
        gru_1 = torch.nn.GRU(self.wv_dim, self.wv_dim, batch_first=True)

        gru_5 = torch.nn.GRU(self.wv_dim, self.wv_dim, batch_first=True)

        conv1 = nn.Conv1d(in_channels=self.wv_dim, out_channels=100,
                          kernel_size=(3, self.wv_dim), stride=(1, self.wv_dim))

        ds3 = nn.Linear(self.wv_dim, self.wv_dim, bias=True)

        att1 = nn.Linear(self.wv_dim, 1, bias=True)
        att3 = nn.Linear(self.wv_dim, 1, bias=True)

        mp1_b = conv1(gru_1(desc1))
        mp2_b = conv1(gru_1(desc2))

        att1_w = torch.softmax(torch.tanh(att1(mp1_b)), dim=-2)
        att2_w = torch.softmax(torch.tanh(att1(mp2_b)), dim=-2)

        size1 = self.default_desc_length

        mp1_b = torch.multiply(mp1_b, size1 * att1_w)
        mp2_b = torch.multiply(mp2_b, size1 * att2_w)

        mp1_b = gru_5(mp1_b)
        mp2_b = gru_5(mp2_b)

        att1_w = torch.softmax(torch.tanh(att3(mp1_b)), dim=-2)
        att2_w = torch.softmax(torch.tanh(att3(mp2_b)), dim=-2)

        mp1_b = torch.multiply(mp1_b, att1_w)
        mp2_b = torch.multiply(mp2_b, att2_w)

        ds1_b = torch.sum(mp1_b, 1).squeeze()
        ds2_b = torch.sum(mp2_b, 1).squeeze()
        eb_desc_batch1 = F.normalize(torch.tanh(ds3(ds1_b)), 2, dim=1)
        eb_desc_batch2 = F.normalize(torch.tanh(ds3(ds2_b)), 2, dim=1)

        indicator = np.empty((self.desc_batch_size, self.desc_batch_size), dtype=np.float32)
        indicator.fill(self.negative_indication_weight)
        np.fill_diagonal(indicator, 1.)
        indicator = to_tensor(indicator)
        desc_loss = -torch.sum(torch.log(torch.sigmoid(torch.multiply(torch.matmul(eb_desc_batch1,
                                                                                   eb_desc_batch2.permute(1, 0)),
                                                                      indicator)) + 0.)) / self.desc_batch_size

        self.desc_embedding1 = eb_desc_batch1
        self.desc_embedding2 = eb_desc_batch2
        return desc_loss, eb_desc_batch1, eb_desc_batch2

    def find_new_alignment_rel(self, un_aligned_ent1, un_aligned_ent2):
        t = time.time()
        embeds1 = self.ent_embeds(un_aligned_ent1).detach().numpy()
        embeds2 = self.ent_embeds(un_aligned_ent2).detach().numpy()
        mapping_mat = self.mapping_mat.detach().numpy()
        embeds1 = np.matmul(embeds1, mapping_mat)
        sim_mat = sim(embeds1, embeds2, normalize=True)
        return sim_mat

        # del dem1, dem2
        # gc.collect()
    def valid(self, stop_metric):
        if len(self.kgs.valid_links) > 0:
            seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities1)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities2)), 2, -1)
        else:
            seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities1)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities2)), 2, -1)
        hits1_12, mrr_12 = valid(seed_entity1.detach().numpy(), seed_entity2.detach().numpy(), self.mapping_matrix.detach().numpy(),
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12

    def tests(self, entities1, entities2):
        seed_entity1 = F.normalize(self.ent_embeds(to_tensor(entities1)), 2, -1)
        seed_entity2 = F.normalize(self.ent_embeds(to_tensor(entities2)), 2, -1)
        _, _, _, sim_list = test(seed_entity1.detach().numpy(), seed_entity2.detach().numpy(), self.mapping_matrix.detach().numpy(),
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric, normalize=self.args.eval_norm,
                                 csls_k=0, accurate=True)
        print()
        return sim_list
