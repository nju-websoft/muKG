import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, empty
from torch.cuda import empty_cache
from torch.nn import Parameter
from tqdm import tqdm

from src.py.load import read
from src.py.util.util import to_var
from src.torch.kge_models.basic_model import BasicModel


class TransH(BasicModel):

    def __init__(self, kgs, args):
        super(TransH, self).__init__(args, kgs)

        self.evaluated_projections = False
        self.dim = self.args.dim
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]),
            requires_grad=False
        )
        self.epsilon = 2.0
        self.p_norm = 1
        self.projected = False
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.norm_vector = nn.Embedding(self.rel_tot, self.dim)
        self.projected_entities = Parameter(empty(size=(self.rel_tot,
                                                        self.ent_tot,
                                                        self.dim)),
                                            requires_grad=False)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
        )
        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        nn.init.uniform_(
            tensor=self.norm_vector.weight.data,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def calc(self, h, r, t):
        '''h = F.normalize(h, 2, -1)
        r = F.normalize(r, 2, -1)
        t = F.normalize(t, 2, -1)'''
        score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1)
        return score

    def evaluate_projections(self):
        """Link prediction evaluation helper function. Project all entities
        according to each relation. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        """
        if self.projected:
            return

        for i in tqdm(range(self.ent_tot), unit='entities', desc='Projecting entities'):

            norm_vect = self.norm_vector.weight.data.view(self.rel_tot, self.dim)
            mask = tensor([i], device=norm_vect.device).long()

            if norm_vect.is_cuda:
                empty_cache()

            ent = self.ent_embeddings(mask)
            norm_components = (ent.view(1, -1) * norm_vect).sum(dim=1)
            self.projected_entities[:, i, :] = (ent.view(1, -1) - norm_components.view(-1, 1) * norm_vect)

            del norm_components

        self.projected = True

    def transfer(self, e, norm):
        norm = F.normalize(norm, p=2, dim=-1)
        return e - (e * norm).sum(dim=1).view(-1, 1) * norm

    def get_score(self, h, r, t):
        return self.calc(h, r, t)

    def get_embeddings(self, hid, rid, tid, mode='entity'):
        h = to_var(hid, self.device)
        r = to_var(rid, self.device)
        t = to_var(tid, self.device)
        self.evaluate_projections()
        r_embs = self.rel_embeddings(r).unsqueeze(1)

        if mode == 'entity':
            proj_h = self.projected_entities[rid, hid].unsqueeze(1)  # shape: (b_size, 1, emb_dim)
            proj_t = self.projected_entities[rid, tid].unsqueeze(1)  # shape: (b_size, 1, emb_dim)
            candidates = self.projected_entities[rid]  # shape: (b_size, self.n_rel, self.emb_dim)
            return proj_h, r_embs, proj_t, candidates
        else:
            proj_h = self.projected_entities[:, hid].transpose(0, 1)  # shape: (b_size, n_rel, emb_dim)
            b_size = proj_h.shape[0]
            proj_t = self.projected_entities[:, tid].transpose(0, 1)  # shape: (b_size, n_rel, emb_dim)
            candidates = self.rel_embeddings.weight.data.view(1, self.rel_tot, self.emb_dim)
            candidates = candidates.expand(b_size, self.rel_tot, self.emb_dim)
            return proj_h, r_embs, proj_t, candidates

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_norm = self.norm_vector(batch_r)
        h = self.transfer(h, r_norm)
        t = self.transfer(t, r_norm)
        score = self.calc(h, r, t).flatten()
        return score

    def predict(self, data):
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()

    def save(self):
        ent_embeds = self.ent_embeddings.cpu().weight.data
        rel_embeds = self.rel_embeddings.cpu().weight.data
        read.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=None)
        norm_vector = self.norm_vector.cpu().weight.data
        read.save_special_embeddings(self.out_folder, 'norm_vector', '', norm_vector, None)
