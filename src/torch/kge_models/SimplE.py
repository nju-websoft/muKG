import torch
import torch.nn as nn

from src.py.util.util import to_var
from src.torch.kge_models.basic_model import BasicModel


class SimplE(BasicModel):

    def __init__(self, args, kgs):
        super(SimplE, self).__init__(args, kgs)

        self.dim = self.args.dim
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.rel_inv_embeddings = nn.Embedding(self.rel_tot, self.dim)
        if self.args.init == 'xavier':
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_inv_embeddings.weight.data)
        else:
            nn.init.xavier_normal_(self.ent_embeddings.weight.data)
            nn.init.xavier_normal_(self.rel_embeddings.weight.data)
            nn.init.xavier_normal_(self.rel_inv_embeddings.weight.data)

    def calc_avg(self, h, t, r, r_inv):
        return (torch.sum(h * r * t, -1) + torch.sum(h * r_inv * t, -1))/2

    def calc_ingr(self, h, r, t):
        return torch.sum(h * r * t, -1)

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_inv = self.rel_inv_embeddings(batch_r)
        score = -self._calc_avg(h, t, r, r_inv)
        return score

    def get_embeddings(self, h_idx, r_idx, t_idx, mode='entities'):
        h_idx = to_var(h_idx, self.device)
        r_idx = to_var(r_idx, self.device)
        t_idx = to_var(t_idx, self.device)
        b_size = h_idx.shape[0]
        h_emb = self.ent_embeddings(h_idx)
        t_emb = self.ent_embeddings(t_idx)
        r_emb = self.rel_embeddings(r_idx)
        r_inv = self.rel_inv_embeddings(r_idx)
        if mode == 'entities':
            candidates = self.ent_embeddings.weight.data.view(1, self.ent_tot, self.dim)
            candidates = candidates.expand(b_size, self.ent_tot, self.dim)
        else:
            candidates = self.rel_embeddings.weight.data.view(1, self.rel_tot, self.dim)
            candidates = candidates.expand(b_size, self.rel_tot, self.dim)
            candidates2 = self.rel_inv_embeddings.weight.data.view(1, self.rel_tot, self.dim)
            candidates2 = candidates2.expand(b_size, self.rel_tot, self.dim)
            candidates = (candidates, candidates2)
        return h_emb, (r_emb, r_inv), t_emb, candidates

    def get_score(self, h, r, t):
        b_size = h.shape[0]
        r_e = r[0]
        r_inv = r[1]
        if len(t.shape) == 3:
            assert (len(h.shape) == 2) & (len(r_e.shape) == 2)
            # this is the tail completion case in link prediction
            hr = (h * r_e).view(b_size, 1, self.dim)
            hr2 = (h * r_inv).view(b_size, 1, self.dim)
            return -(hr * t).sum(dim=2) - (hr2 * t).sum(dim=2)
        elif len(h.shape) == 3:
            assert (len(t.shape) == 2) & (len(r_e.shape) == 2)
            # this is the head completion case in link prediction
            rt = (r_e * t).view(b_size, 1, self.dim)
            rt2 = (r_inv * t).view(b_size, 1, self.dim)
            return -(h * rt).sum(dim=2) - (h * rt2).sum(dim=2)
        elif len(r.shape) == 3:
            assert (len(h.shape) == 2) & (len(t.shape) == 2)
            # this is the relation prediction case
            hr = (h.view(b_size, 1, self.dim) * r_e)  # hr has shape (b_size, self.n_rel, self.emb_dim)
            hr2 = (h.view(b_size, 1, self.dim) * r_inv)
            return -(hr * t.view(b_size, 1, self.dim)).sum(dim=2) - (hr2 * t.view(b_size, 1, self.dim)).sum(dim=2)