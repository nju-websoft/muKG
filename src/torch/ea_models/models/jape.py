import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.py.evaluation.evaluation import valid, test
from src.py.util.util import to_tensor
from src.torch.kge_models.basic_model import BasicModel


class JAPE(BasicModel):

    def __init__(self, args, kgs):
        super(JAPE, self).__init__(args, kgs)
        # self.attr2vec = Attr2Vec()
        self.attr_sim_mat_place = None
        self.entities1 = None
        self.attr_sim_mat = None
        self.ref_entities1, self.ref_entities2 = None, None
        
    def init(self):
        self.ref_entities1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_entities2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        self._define_variables()
        # customize parameters
        assert self.args.alignment_module == 'sharing'
        assert self.args.init == 'normal'
        assert self.args.neg_sampling == 'uniform'
        #assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        assert self.args.loss_norm == 'L2'

        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True

        assert self.args.neg_triple_num >= 1
        assert self.args.neg_alpha >= 0.0
        assert self.args.top_attr_threshold > 0.0
        assert self.args.attr_sim_mat_threshold > 0.0
        assert self.args.attr_sim_mat_beta > 0.0

    def _define_variables(self):
        self.ent_embeds = nn.Embedding(self.kgs.entities_num, self.args.dim)
        self.rel_embeds = nn.Embedding(self.kgs.relations_num, self.args.dim)
        self.margin = nn.Parameter(torch.Tensor([1]))
        self.margin.requires_grad = False
        nn.init.xavier_uniform_(self.ent_embeds.weight.data)
        nn.init.xavier_uniform_(self.rel_embeds.weight.data)
        #self.ent_embeds.weight.data = F.normalize(self.ent_embeds.weight.data, 2, -1)
        #self.rel_embeds.weight.data = F.normalize(self.rel_embeds.weight.data, 2, -1)
        '''if self.args.init == 'xavier':
            nn.init.xavier_uniform_(self.ent_embeds.weight.data)
            nn.init.xavier_uniform_(self.rel_embeds.weight.data)
        elif self.args.init == 'normal':
            std = 1.0 / math.sqrt(self.args.dim)
            nn.init.normal_(self.ent_embeds.weight.data, 0, std)
            nn.init.normal_(self.rel_embeds.weight.data, 0, std)
        elif self.args.init == 'uniform':
            nn.init.uniform_(self.ent_embeds.weight.data, 0)
            nn.init.uniform_(self.rel_embeds.weight.data, 0)'''

    def define_embed_graph(self, data):
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
        if self.args.loss_norm == "L2":
            pos = torch.pow(torch.norm(ph + pr - pt, 2, -1), 2)
            neg = torch.pow(torch.norm(nh + nr - nt, 2, -1), 2)
        else:
            pos = torch.norm(ph + pr - pt, 1, -1)
            neg = torch.norm(nh + nr - nt, 1, -1)
        pos = pos.view(batch_size_now, -1)
        neg = neg.view(batch_size_now, -1)
        return torch.sum(torch.relu_(pos - neg + self.margin))

    def valid(self, stop_metric):
        if len(self.kgs.valid_links) > 0:
            seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities1, self.device)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities2 + self.kgs.test_entities2, self.device)), 2, -1)
        else:
            seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities1, self.device)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities2, self.device)), 2, -1)
        hits1_12, mrr_12 = valid(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(),
                                 None, self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12

    def define_sim_graph(self, data):
        entities1 = data['entities1']
        attr_sim_mat_place = data['sim_mat']
        ref1 = F.normalize(self.ent_embeds(entities1), 2, -1)
        ref2 = self.ent_embeds(to_tensor(self.ref_entities2, self.device))
        ref2_trans = torch.matmul(attr_sim_mat_place, ref2)
        ref2_trans = F.normalize(ref2_trans, 2, -1)
        sim_loss = self.args.attr_sim_mat_beta * torch.sum(torch.pow(torch.norm(ref1 - ref2_trans), 2))
        return sim_loss

    def tests(self, entities1, entities2):
        seed_entity1 = F.normalize(self.ent_embeds(to_tensor(entities1, self.device)), 2, -1)
        seed_entity2 = F.normalize(self.ent_embeds(to_tensor(entities2, self.device)), 2, -1)
        _, _, _, sim_list = test(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(),
                                 None,
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm,
                                 csls_k=0, accurate=True)
        print()
        return sim_list

