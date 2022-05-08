import gc
import math
import random
import time

import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.py.evaluation.evaluation import valid, test
from src.py.util.util import to_tensor
from src.torch.kge_models.basic_model import BasicModel


def early_stop(flag1, flag2, flag):
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False


def add_compositional_func(character_vectors):
    value_vector_list = torch.mean(character_vectors, 1).squeeze()
    value_vector_list = F.normalize(value_vector_list, 2, -1)
    return value_vector_list


def n_gram_compositional_func(character_vectors, value_lens, batch_size, embed_size):
    # pos_c_e_in_lstm = tf.unstack(character_vectors, num=value_lens, axis=1)
    pos_c_e_in_lstm = character_vectors
    pos_c_e_lstm = calculate_ngram_weight(pos_c_e_in_lstm, value_lens, batch_size, embed_size)
    return pos_c_e_lstm


def calculate_ngram_weight(unstacked_tensor, value_lens, batch_size, embed_size):
    stacked_tensor = unstacked_tensor
    stacked_tensor = torch.flip(stacked_tensor, [1])
    index = value_lens - 1
    expected_result = torch.zeros([batch_size, embed_size])
    while index >= 0:
        precessed = stacked_tensor[:, index:, :]
        summand = torch.mean(precessed, dim=1).squeeze()
        expected_result += summand
        index = index - 1
    return expected_result


class AttrE(BasicModel):

    def __init__(self, args, kgs):
        super(AttrE, self).__init__(args, kgs)
        self.char_list_size = None
        self.value_id_char_ids = None
        self.attribute_triples_list2 = None
        self.attribute_triples_list1 = None

    def initial(self, char_list_size):
        '''self.attribute_triples_list1, self.attribute_triples_list2, self.value_id_char_ids, self.char_list_size = \
            formatting_attr_triples(self.kgs, self.args.literal_len)'''
        self.char_list_size = char_list_size
        self._define_variables()

    def _define_variables(self):
        self.ent_embeds = nn.Embedding(self.kgs.entities_num, self.args.dim)
        self.rel_embeds = nn.Embedding(self.kgs.relations_num, self.args.dim)
        self.ent_embeds_ce = nn.Embedding(self.kgs.entities_num, self.args.dim)
        self.attr_embeds = nn.Embedding(self.kgs.attributes_num, self.args.dim)
        self.char_embeds = nn.Embedding(self.char_list_size, self.args.dim)
        '''self.ent_embeds = nn.Parameter(torch.Tensor(self.kgs.entities_num, self.args.dim))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.kgs.relations_num, self.args.dim))
        self.ent_embeds_ce = nn.Parameter(torch.Tensor(self.kgs.entities_num, self.args.dim))
        self.attr_embeds = nn.Parameter(torch.Tensor(self.kgs.attributes_num, self.args.dim))
        self.char_embeds = nn.Parameter(torch.Tensor(self.char_list_size, self.args.dim))'''
        self.margin = nn.Parameter(torch.Tensor([1.5]))
        self.margin.requires_grad = False
        nn.init.xavier_normal_(self.ent_embeds.weight.data)
        nn.init.xavier_normal_(self.rel_embeds.weight.data)
        nn.init.xavier_normal_(self.ent_embeds_ce.weight.data)
        nn.init.xavier_normal_(self.attr_embeds.weight.data)
        nn.init.xavier_normal_(self.char_embeds.weight.data)
        '''norms = torch.norm(self.ent_embeds.weight, p=2, dim=1).data
        self.ent_embeds.weight.data = self.ent_embeds.weight.data.div(
            norms.view(self.kgs.entities_num, 1).expand_as(self.ent_embeds.weight))

        norms = torch.norm(self.rel_embeds.weight, p=2, dim=1).data
        self.rel_embeds.weight.data = self.rel_embeds.weight.data.div(
            norms.view(self.kgs.relations_num, 1).expand_as(self.rel_embeds.weight))

        norms = torch.norm(self.ent_embeds_ce.weight, p=2, dim=1).data
        self.ent_embeds_ce.weight.data = self.ent_embeds_ce.weight.data.div(
            norms.view(self.kgs.entities_num, 1).expand_as(self.ent_embeds_ce.weight))

        norms = torch.norm(self.attr_embeds.weight, p=2, dim=1).data
        self.attr_embeds.weight.data = self.attr_embeds.weight.data.div(
            norms.view(self.kgs.attributes_num, 1).expand_as(self.attr_embeds.weight))

        norms = torch.norm(self.char_embeds.weight, p=2, dim=1).data
        self.char_embeds.weight.data = self.char_embeds.weight.data.div(
            norms.view(self.char_list_size, 1).expand_as(self.char_embeds.weight))'''
        # self.ent_embeds.weight.data = F.normalize(self.ent_embeds.weight.data, 2, -1)
        # self.rel_embeds.weight.data = F.normalize(self.rel_embeds.weight.data, 2, -1)
        # self.ent_embeds_ce.weight.data = F.normalize(self.ent_embeds_ce.weight.data, 2, -1)
        # self.attr_embeds.weight.data = F.normalize(self.attr_embeds.weight.data, 2, -1)
        # self.char_embeds.weight.data = F.normalize(self.char_embeds.weight.data, 2, -1)

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
        """
        ph = self.ent_embeds(ph)
        pr = self.rel_embeds(pr)
        pt = self.ent_embeds(pt)
        nh = self.ent_embeds(nh)
        nr = self.rel_embeds(nr)
        nt = self.ent_embeds(nt)
        """
        pos = torch.pow(torch.norm(ph + pr - pt, 2, -1), 2)
        neg = torch.pow(torch.norm(nh + nr - nt, 2, -1), 2)
        pos = pos.view(batch_size_now, -1)
        neg = neg.view(batch_size_now, -1)
        return torch.sum(torch.relu_(pos - neg + self.margin))

    def generate_attribute_loss(self, data):
        pes = data['pos_hs']
        pas = data['pos_rs']
        pvs = data['pos_ts']
        nes = data['neg_hs']
        nas = data['neg_rs']
        nvs = data['neg_ts']
        batch_size_now = pes.shape[0]
        ph = F.normalize(self.ent_embeds_ce(pes), 2, -1)
        pr = F.normalize(self.attr_embeds(pas), 2, -1)
        pt = F.normalize(self.char_embeds(pvs), 2, -1)
        nh = F.normalize(self.ent_embeds_ce(nes), 2, -1)
        nr = F.normalize(self.attr_embeds(nas), 2, -1)
        nt = F.normalize(self.char_embeds(nvs), 2, -1)
        """
        ph = self.ent_embeds_ce(pes)
        pr = self.attr_embeds(pas)
        pt = self.char_embeds(pvs)
        nh = self.ent_embeds_ce(nes)
        nr = self.attr_embeds(nas)
        nt = self.char_embeds(nvs)
        """
        '''pt = n_gram_compositional_func(pt, self.args.literal_len, batch_size_now, self.args.dim)
        nt = n_gram_compositional_func(nt, self.args.literal_len,
                                       batch_size_now, self.args.dim)'''
        pt = add_compositional_func(pt)
        nt = add_compositional_func(nt)
        '''pt = F.normalize(pt, 2, -1)
        nt = F.normalize(nt, 2, -1)'''''
        if self.args.loss_norm == "L2":
            pos = torch.pow(torch.norm(ph + pr - pt, 2, -1), 2)
            neg = torch.pow(torch.norm(nh + nr - nt, 2, -1), 2)
        else:
            pos = torch.norm(ph + pr - pt, 1, -1)
            neg = torch.norm(nh + nr - nt, 1, -1)
        pos = pos.view(batch_size_now, -1)
        neg = neg.view(batch_size_now, -1)
        return torch.sum(torch.relu_(pos - neg + self.margin))

    def generate_align_loss(self, data):
        ent_se = self.ent_embeds(data)
        ent_ce = self.ent_embeds_ce(data)
        '''norm1 = torch.pow(torch.norm(ent_se, 2, -1), 2)
        norm2 = torch.pow(torch.norm(ent_ce, 2, -1), 2)
        norm = torch.mul(norm2, norm1)
        cos_sim = torch.sum(torch.multiply(ent_se, ent_ce), 1).div(norm)
        loss = torch.sum(torch.relu_(1 - cos_sim))'''
        loss = torch.sum(torch.pow(torch.norm(ent_se - ent_ce, 2, -1), 2))
        return loss

    def test(self, entities1, entities2):
        seed_entity1 = F.normalize(self.ent_embeds(entities1), 2, -1)
        seed_entity2 = F.normalize(self.ent_embeds(entities2), 2, -1)
        """
        seed_entity1 = self.ent_embeds(entities1)
        seed_entity2 = self.ent_embeds(entities2)
        """
        _, _, _, sim_list = test(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(), None,
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm,
                                 csls_k=0, accurate=True)
        print()
        return sim_list

