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

import src.modules.finding.evaluation as evaluation
import src.modules.load.read as read
from src.loss.nce_loss import NCELoss
from src.models.basic_model import BasicModel
from src.modules.load.kg import KG
from src.modules.load.kgs import KGs
from src.trainer.util import get_optimizer, to_tensor
import pandas as pd
from scipy.sparse import csr_matrix


class RSN4EA(BasicModel):
    def __init__(self, kgs, args):
        super(RSN4EA, self).__init__(args, kgs)
        self._train_data = None
        self._options = None
        self._r_m = None
        self._e_m = None
        self._kb = None
        self.criterion = nn.LogSigmoid()
        self._rel_testing = None
        self._ent_testing = None
        self._rel_mapping = None
        self._ent_mapping = None
        self._rel_num = None
        self._ent_num = None

    def initial(self, ent_num, rel_num):
        self._options = opts = self.args
        self._ent_num = ent_num
        self._rel_num = rel_num
        self._define_variables()

    def _define_variables(self):
        options = self._options
        hidden_size = options.hidden_size
        self._entity_embedding = nn.Embedding(self._ent_num, hidden_size)
        self._relation_embedding = nn.Embedding(self._rel_num, hidden_size)
        self._rel_w = nn.Embedding(self._rel_num, hidden_size)
        self._ent_w = nn.Embedding(self._ent_num, hidden_size)
        self._rel_b = nn.Parameter(torch.zeros(self._rel_num))
        self._ent_b = nn.Parameter(torch.zeros(self._ent_num))
        nn.init.xavier_uniform_(self._entity_embedding.weight.data)
        nn.init.xavier_uniform_(self._relation_embedding.weight.data)
        nn.init.xavier_uniform_(self._rel_w.weight.data)
        nn.init.xavier_uniform_(self._ent_w.weight.data)
        self.l1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.l2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rnn = nn.RNN(
            input_size=hidden_size,
            hidden_size=options.hidden_size,
            num_layers=options.num_layers,
            batch_first=True,
            dropout=options.keep_prob,
            # nonlinearity=nn.Identity()
        )
        self.ent_embeds, self.rel_embeds = self._entity_embedding, self._relation_embedding
        self.bn = nn.BatchNorm1d(hidden_size)
        self.entity_w, self._entity_b = self._ent_w, self._ent_b

    def sampled_loss_rel(self, inputs, labels, weight=1):
        num_sampled = min(self._options.num_samples, self._rel_num // 3)
        labels = labels.view(-1, 1)
        batch_size = inputs.shape[0]
        result = []
        num = list(range(self._rel_num))
        neg = to_tensor(random.sample(num, batch_size * inputs.shape[1]))
        # neg_attr1 = neg_attr1.view(batch_size, -1, 1)
        neg = neg.view(batch_size, -1)
        pos2 = self._rel_w(labels).permute(0, 2, 1)
        neg2 = self._rel_w(neg).permute(0, 2, 1)
        pos = torch.bmm(inputs, pos2).squeeze(2).flatten()
        neg = torch.bmm(inputs, neg2.neg()).squeeze(1).flatten()
        losses = -(self.criterion(pos) + self.criterion(neg)) / 2
        return losses * weight

    def sampled_loss_entity(self, inputs, labels, weight=1):
        num_sampled = min(self._options.num_samples, self._ent_num // 3)
        labels = labels.view(-1, 1)
        batch_size = inputs.shape[0]
        result = []
        num = list(range(self._ent_num))
        neg = to_tensor(random.sample(num, batch_size * inputs.shape[1]))
        # neg_attr1 = neg_attr1.view(batch_size, -1, 1)
        neg = neg.view(batch_size, -1)
        pos2 = self._ent_w(labels).permute(0, 2, 1)
        neg2 = self._ent_w(neg).permute(0, 2, 1)
        pos = torch.bmm(inputs, pos2).squeeze(2).flatten()
        neg = torch.bmm(inputs, neg2.neg()).squeeze(1).flatten()
        losses = -(self.criterion(pos) + self.criterion(neg)) / 2
        return losses * weight

    # shuffle data
    def sample(self, data):
        choices = np.random.choice(len(data), size=len(data), replace=False)
        return data.iloc[choices]

    # build an RSN of length l
    def build_sub_graph(self, length, reuse, data):
        options = self._options
        hidden_size = options.hidden_size
        batch_size = options.batch_size
        seq = data['seq']

        e_em, r_em = self._entity_embedding, self._relation_embedding

        # seperately read, and then recover the order
        ent = seq[:, :-1:2]
        rel = seq[:, 1::2]

        ent_em = self._entity_embedding(ent)
        rel_em = self._relation_embedding(rel)

        em_seq = []
        for i in range(length - 1):
            if i % 2 == 0:
                em_seq.append(ent_em[:, i // 2])
            else:
                em_seq.append(rel_em[:, i // 2])

        # seperately bn
        if not reuse:
            bn_em_seq = [self.bn(em_seq[i]).view(-1, 1, hidden_size) for i in range(length - 1)]
        else:
            bn_em_seq = [self.bn(em_seq[i]).view(-1, 1, hidden_size) for i in range(length - 1)]

        bn_em_seq = torch.cat(bn_em_seq, dim=1)

        ent_bn_em = bn_em_seq[:, ::2]

        outputs, state = self.rnn(bn_em_seq)

        # with tf.variable_scope('transformer', reuse=reuse):
        #     outputs = transformer_model(input_tensor=bn_em_seq,
        #                                 hidden_size=hidden_size,
        #                                 intermediate_size=hidden_size*4,
        #                                 num_attention_heads=8)

        rel_outputs = outputs[:, 1::2, :]
        outputs = [outputs[:, i, :] for i in range(length - 1)]

        ent_outputs = outputs[::2]

        # RSN
        res_rel_outputs = self.l1(rel_outputs) + self.l2(ent_bn_em)

        # recover the order
        res_rel_outputs = [res_rel_outputs[:, i, :] for i in range((length - 1) // 2)]
        outputs = []
        for i in range(length - 1):
            if i % 2 == 0:
                outputs.append(ent_outputs[i // 2])
            else:
                outputs.append(res_rel_outputs[i // 2])

        # output bn
        if reuse:
            bn_outputs = [self.bn(outputs[i]).view(-1, 1, hidden_size) for i in range(length - 1)]
        else:
            bn_outputs = [self.bn(outputs[i]).view(-1, 1, hidden_size) for i in range(length - 1)]

        def cal_loss(bn_outputs, seq):
            losses = []

            masks = np.random.choice([0., 1.0], size=batch_size, p=[0.5, 0.5])
            weight = to_tensor(masks)
            for i, output in enumerate(bn_outputs):
                if i % 2 == 0:
                    losses.append(self.sampled_loss_rel(
                        output, seq[:, i + 1], weight))
                else:
                    losses.append(self.sampled_loss_entity(
                        output, seq[:, i + 1], weight))
            losses = torch.stack(losses, dim=1)
            return losses

        seq_loss = cal_loss(bn_outputs, seq)

        losses = torch.sum(seq_loss).squeeze() / batch_size

        return losses

    # build the main graph
    def tests(self, entities1, entities2):
        seed_entity1 = self.ent_embeds(to_tensor(entities1))
        seed_entity2 = self.ent_embeds(to_tensor(entities2))
        _, _, _, sim_list = evaluation.test(seed_entity1.detach().numpy(), seed_entity2.detach().numpy(), None,
                                            self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                                            normalize=self.args.eval_norm,
                                            csls_k=0, accurate=True)
        print()
        return sim_list

    def valid(self, stop_metric):
        if len(self.kgs.valid_links) > 0:
            seed_entity1 = self.ent_embeds(to_tensor(self.kgs.valid_entities1))
            seed_entity2 = self.ent_embeds(to_tensor(self.kgs.valid_entities2))
        else:
            seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities1)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities2)), 2, -1)
        hits1_12, mrr_12 = evaluation.valid(seed_entity1.detach().numpy(), seed_entity2.detach().numpy(), None,
                                            self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                                            normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12
    # training procedure
