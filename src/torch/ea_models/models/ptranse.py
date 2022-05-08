import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.py.evaluation.evaluation import test, valid
from src.py.util.util import to_tensor
from src.torch.kge_models.basic_model import BasicModel


class IPTransE(BasicModel):

    def __init__(self, args, kgs):
        super().__init__(args, kgs)
        self.ref_entities2 = None
        self.ref_entities1 = None
        # self.out_folder = r'D:\OPENEA-pytorch\result'

    def init(self):
        self.ref_entities1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_entities2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        self._define_variables()
        # customize parameters
        assert self.args.alignment_module == 'sharing'
        #assert self.args.init == 'normal'
        assert self.args.neg_sampling == 'uniform'
        assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        assert self.args.loss_norm == 'L2'
        #self.args.loss_norm == 'L2'
        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True

        assert self.args.margin > 0.0
        #assert self.args.neg_triple_num == 1
        assert self.args.sim_th > 0.0

    def _define_variables(self):
        self.ent_embeds = nn.Embedding(self.kgs.entities_num, self.args.dim)
        self.rel_embeds = nn.Embedding(self.kgs.relations_num, self.args.dim)

        self.margin = nn.Parameter(torch.Tensor([self.args.margin]))
        self.margin.requires_grad = False
        if self.args.init == 'xavier':
            nn.init.xavier_uniform_(self.ent_embeds.weight.data)
            nn.init.xavier_uniform_(self.rel_embeds.weight.data)
        elif self.args.init == 'normal':
            std = 1.0 / math.sqrt(self.args.dim)
            nn.init.trunc_normal_(self.ent_embeds.weight.data, std=std)
            nn.init.trunc_normal_(self.rel_embeds.weight.data, std=std)
        elif self.args.init == 'uniform':
        		self.embedding_range = nn.Parameter(
        			torch.Tensor([(self.margin.item() + self.epsilon) / self.dim]), 
        			requires_grad=False
        		)
        		nn.init.uniform_(tensor=self.ent_embeds.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        		nn.init.uniform_(tensor=self.rel_embeds.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        self.ent_embeds.weight.data = F.normalize(self.ent_embeds.weight.data, 2, -1)
        self.rel_embeds.weight.data = F.normalize(self.rel_embeds.weight.data, 2, -1)

    def ref_sim_mat(self):
        ref1_embeddings = self.ent_embeds(to_tensor(self.ref_entities1, self.device)).cpu().detach().numpy()
        ref2_embeddings = self.ent_embeds(to_tensor(self.ref_entities2, self.device)).cpu().detach().numpy()
        sim_mat = np.matmul(ref1_embeddings, ref2_embeddings.T)
        return sim_mat

    def generate_transE_loss(self, data):
        phs = data['pos_hs']
        prs = data['pos_rs']
        pts = data['pos_ts']
        nhs = data['neg_hs']
        nrs = data['neg_rs']
        nts = data['neg_ts']
        prx = data['pos_rx']
        pry = data['pos_ry']
        pr = data['pos_r']
        nrx = data['neg_rx']
        nry = data['neg_ry']
        nr = data['neg_r']
        ws = data['path_weight']
        batch_size_now = phs.shape[0]
        phs = self.ent_embeds(phs)
        prs = self.rel_embeds(prs)
        pts = self.ent_embeds(pts)
        nhs = self.ent_embeds(nhs)
        nrs = self.rel_embeds(nrs)
        nts = self.ent_embeds(nts)
        prx = self.rel_embeds(prx)
        pry = self.rel_embeds(pry)
        pr = self.rel_embeds(pr)
        nrx = self.rel_embeds(nrx)
        nry = self.rel_embeds(nry)
        nr = self.rel_embeds(nr)
        if self.args.loss_norm == "L2":
            pos = torch.pow(torch.norm(phs + prs - pts, 2, -1), 2)
            neg = torch.pow(torch.norm(nhs + nrs - nts, 2, -1), 2)
        else:
            pos = torch.pow(torch.norm(phs + prs - pts, 1, -1), 2)
            neg = torch.pow(torch.norm(nhs + nrs - nts, 1, -1), 2)
        pos1 = pos.view(batch_size_now, -1)
        neg1 = neg.view(batch_size_now, -1)
        ptranse_loss1 = torch.sum(torch.relu_(pos1 - neg1 + self.margin))
        pos2 = torch.pow(torch.norm(prx + pry - pr, 2, -1), 2)
        neg2 = torch.pow(torch.norm(nrx + nry - nr, 2, -1), 2)
        ws = 1 / ws
        ptranse_loss2 = torch.sum(ws * torch.relu_(pos2 - neg2 + self.margin))
        return ptranse_loss1 + self.args.path_parm * ptranse_loss2

    def generate_align_loss(self, data):
        phs = data['pos_hs']
        prs = data['pos_rs']
        pts = data['pos_ts']
        nhs = data['neg_hs']
        nrs = data['neg_rs']
        nts = data['neg_ts']
        ws = data['path_weight']
        batch_size_now = phs.shape[0]
        phs = self.ent_embeds(phs)
        prs = self.rel_embeds(prs)
        pts = self.ent_embeds(pts)
        nhs = self.ent_embeds(nhs)
        nrs = self.rel_embeds(nrs)
        nts = self.ent_embeds(nts)
        pos = torch.pow(torch.norm(phs + prs - pts, 2, -1), 2)
        neg = torch.pow(torch.norm(nhs + nrs - nts, 2, -1), 2)
        return torch.sum(ws * torch.relu_(pos + self.args.margin - neg))

    def tests(self, entities1, entities2):
        seed_entity1 = self.ent_embeds(to_tensor(entities1, self.device))
        seed_entity2 = self.ent_embeds(to_tensor(entities2, self.device))
        _, _, _, sim_list = test(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(), None,
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric, normalize=self.args.eval_norm,
                                 csls_k=0, accurate=True)
        print()
        return sim_list

    def valid(self, stop_metric):
        if len(self.kgs.valid_links) > 0:
            seed_entity1 = self.ent_embeds(to_tensor(self.kgs.valid_entities1, self.device))
            seed_entity2 = self.ent_embeds(to_tensor(self.kgs.valid_entities2, self.device))
        else:
            seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities1, self.device)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities2, self.device)), 2, -1)
        hits1_12, mrr_12 = valid(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(), None,
                                 self.args.top_k,self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12

"""
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.py.evaluation.evaluation import test, valid
from src.py.util.util import to_tensor
from src.torch.kge_models.basic_model import BasicModel


class IPTransE(BasicModel):

    def __init__(self, args, kgs):
        super().__init__(args, kgs)
        self.ref_entities2 = None
        self.ref_entities1 = None
        # self.out_folder = r'D:\OPENEA-pytorch\result'

    def init(self):
        self.ref_entities1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_entities2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        self._define_variables()
        # customize parameters
        assert self.args.alignment_module == 'sharing'
        #assert self.args.init == 'normal'
        assert self.args.neg_sampling == 'uniform'
        assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        assert self.args.loss_norm == 'L2'

        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True

        assert self.args.margin > 0.0
        #assert self.args.neg_triple_num == 1
        assert self.args.sim_th > 0.0

    def _define_variables(self):
        self.ent_embeds = nn.Embedding(self.kgs.entities_num, self.args.dim)
        self.rel_embeds = nn.Embedding(self.kgs.relations_num, self.args.dim)

        self.margin = nn.Parameter(torch.Tensor([self.args.margin]))
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
        self.ent_embeds.weight.data = F.normalize(self.ent_embeds.weight.data, 2, -1)
        self.rel_embeds.weight.data = F.normalize(self.rel_embeds.weight.data, 2, -1)

    def ref_sim_mat(self):
        ref1_embeddings = self.ent_embeds(to_tensor(self.ref_entities1, self.device)).cpu().detach().numpy()
        ref2_embeddings = self.ent_embeds(to_tensor(self.ref_entities2, self.device)).cpu().detach().numpy()
        sim_mat = np.matmul(ref1_embeddings, ref2_embeddings.T)
        return sim_mat

    def generate_transE_loss(self, data):
        phs = data['pos_hs']
        prs = data['pos_rs']
        pts = data['pos_ts']
        nhs = data['neg_hs']
        nrs = data['neg_rs']
        nts = data['neg_ts']
        prx = data['pos_rx']
        pry = data['pos_ry']
        pr = data['pos_r']
        nrx = data['neg_rx']
        nry = data['neg_ry']
        nr = data['neg_r']
        ws = data['path_weight']
        batch_size_now = phs.shape[0]
        phs = self.ent_embeds(phs)
        prs = self.rel_embeds(prs)
        pts = self.ent_embeds(pts)
        nhs = self.ent_embeds(nhs)
        nrs = self.rel_embeds(nrs)
        nts = self.ent_embeds(nts)
        prx = self.rel_embeds(prx)
        pry = self.rel_embeds(pry)
        pr = self.rel_embeds(pr)
        nrx = self.rel_embeds(nrx)
        nry = self.rel_embeds(nry)
        nr = self.rel_embeds(nr)
        if self.args.loss_norm == "L2":
            pos = torch.pow(torch.norm(phs + prs - pts, 2, -1), 2)
            neg = torch.pow(torch.norm(nhs + nrs - nts, 2, -1), 2)
        else:
            pos = torch.norm(phs + prs - pts, 1, -1)
            neg = torch.norm(nhs + nrs - nts, 1, -1)
        pos1 = pos.view(batch_size_now, -1)
        neg1 = neg.view(batch_size_now, -1)
        ptranse_loss1 = torch.sum(torch.relu_(pos1 - neg1 + self.margin))
        pos2 = torch.pow(torch.norm(prx + pry - pr, 2, -1), 2)
        neg2 = torch.pow(torch.norm(nrx + nry - nr, 2, -1), 2)
        ws = 1 / ws
        ptranse_loss2 = torch.sum(ws * torch.relu_(pos2 - neg2 + self.margin))
        return ptranse_loss1 + self.args.path_parm * ptranse_loss2

    def generate_align_loss(self, data):
        phs = data['pos_hs']
        prs = data['pos_rs']
        pts = data['pos_ts']
        nhs = data['neg_hs']
        nrs = data['neg_rs']
        nts = data['neg_ts']
        ws = data['path_weight']
        batch_size_now = phs.shape[0]
        phs = self.ent_embeds(phs)
        prs = self.rel_embeds(prs)
        pts = self.ent_embeds(pts)
        nhs = self.ent_embeds(nhs)
        nrs = self.rel_embeds(nrs)
        nts = self.ent_embeds(nts)
        pos = torch.pow(torch.norm(phs + prs - pts, 2, -1), 2)
        neg = torch.pow(torch.norm(nhs + nrs - nts, 2, -1), 2)
        return torch.sum(ws * torch.relu_(pos + self.args.margin - neg))

    def tests(self, entities1, entities2):
        seed_entity1 = self.ent_embeds(to_tensor(entities1, self.device))
        seed_entity2 = self.ent_embeds(to_tensor(entities2, self.device))
        _, _, _, sim_list = test(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(), None,
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric, normalize=self.args.eval_norm,
                                 csls_k=0, accurate=True)
        print()
        return sim_list

    def valid(self, stop_metric):
        if len(self.kgs.valid_links) > 0:
            seed_entity1 = self.ent_embeds(to_tensor(self.kgs.valid_entities1, self.device))
            seed_entity2 = self.ent_embeds(to_tensor(self.kgs.valid_entities2, self.device))
        else:
            seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities1, self.device)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities2, self.device)), 2, -1)
        hits1_12, mrr_12 = valid(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(), None,
                                 self.args.top_k,self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12

"""