import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.py.load import read
from src.py.util.util import to_var
from src.torch.kge_models.basic_model import BasicModel


class TransE_ET(BasicModel):

    def __init__(self, kgs, args, dim=100, p_norm=2, norm_flag=True, margin=1.5, epsilon=None):
        super(TransE_ET, self).__init__(args, kgs)
        self.dim = self.args.dim
        self.margin = 2
        # self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = 1

        '''self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        self.margin_flag = False'''
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.dim]),
            requires_grad=False
        )

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(1, self.dim)


        self.type_embeddings = nn.Embedding(3851, self.dim)
        nn.init.uniform_(tensor=self.ent_embeddings.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_embeddings.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.type_embeddings.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        '''self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, 2, -1)
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, 2, -1)'''


    def calc(self, h, r, t):
        '''h = F.normalize(h, 2, -1)
        r = F.normalize(r, 2, -1)
        t = F.normalize(t, 2, -1)'''
        score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1)
        return score


    def get_embeddings(self, hid, rid, tid):
        h = to_var(hid, self.device)
        r = to_var(0, self.device)
        t = to_var(tid, self.device)
        r_embs = self.rel_embeddings(r).view(1, 1, -1)
        proj_h = self.ent_embeddings(h).unsqueeze(1)  # shape: (b_size, 1, emb_dim)
        b_size = proj_h.shape[0]
        r_embs = r_embs.expand(b_size, 1, -1)
        proj_t = self.type_embeddings(t).unsqueeze(1)  # shape: (b_size, 1, emb_dim)
        candidates = self.type_embeddings.weight.data.view(1, 3851, self.dim)  # shape: (b_size, self.n_rel, self.emb_dim)
        candidates = candidates.expand(b_size, 3851, self.dim)
        return proj_h, r_embs, proj_t, candidates


    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.type_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self.calc(h, r, t).flatten()
        return score

    def save(self):
        ent_embeds = self.ent_embeddings.cpu().weight.data
        rel_embeds = self.rel_embeddings.cpu().weight.data
        read.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=None)
        type_embeddings = self.type_embeddings.cpu().weight.data
        read.save_special_embeddings(self.out_folder, 'type_embeddings', '', type_embeddings, None)
