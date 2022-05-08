import math

import torch
import torch.nn as nn
# from .Model import Model
# from src.models.basic_model import BasicModel
from src.py.load import read
from src.py.util.util import to_var
from src.torch.kge_models.basic_model import BasicModel


class ComplEx(BasicModel):
    def __init__(self, kgs, args, dim=200):
        super(ComplEx, self).__init__(args, kgs)

        self.dim = dim
        self.ent_re_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.ent_im_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_re_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.rel_im_embeddings = nn.Embedding(self.rel_tot, self.dim)
        if self.args.init == 'xavier':
            nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
            nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
        else:
            std = 1.0 / math.sqrt(self.args.dim)
            nn.init.normal_(self.ent_re_embeddings.weight.data, 0, std)
            nn.init.normal_(self.ent_im_embeddings.weight.data, 0, std)
            nn.init.normal_(self.rel_re_embeddings.weight.data, 0, std)
            nn.init.normal_(self.rel_im_embeddings.weight.data, 0, std)

    def calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return -(h_re * (r_re * t_re + r_im * t_im) + h_im * (
                r_re * t_im - r_im * t_re)).sum(dim=1)

    def get_score(self, h, r, t):
        re_h, im_h = h[0], h[1]
        re_t, im_t = t[0], t[1]
        re_r, im_r = r[0], r[1]
        b_size = re_h.shape[0]

        if len(re_t.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_r.shape) == 2)
            # this is the tail completion case in link prediction
            return -((re_h * re_r - im_h * im_r).view(b_size, 1, self.dim) * re_t
                     + (re_h * im_r + im_h * re_r).view(b_size, 1, self.dim) * im_t).sum(dim=2)

        elif len(re_h.shape) == 3:
            assert (len(re_t.shape) == 2) & (len(re_r.shape) == 2)
            # this is the head completion case in link prediction

            return -(re_h * (re_r * re_t + im_r * im_t).view(b_size, 1, self.dim)
                     + im_h * (re_r * im_t - im_r * re_t).view(b_size, 1, self.dim)).sum(dim=2)

        elif len(re_r.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_t.shape) == 2)
            # this is the relation prediction case
            return -((re_h * re_t + im_h * im_t).view(b_size, 1, self.dim) * re_r
                     + (re_h * im_t - im_h * re_t).view(b_size, 1, self.dim) * im_r).sum(dim=2)

    def get_embeddings(self, h_id, r_id, t_id, mode='entities'):
        h_id = to_var(h_id, self.device)
        r_id = to_var(r_id, self.device)
        t_id = to_var(t_id, self.device)
        re_h, im_h = self.ent_re_embeddings(h_id), self.ent_im_embeddings(h_id)
        re_t, im_t = self.ent_re_embeddings(t_id), self.ent_im_embeddings(t_id)
        re_r, im_r = self.rel_re_embeddings(r_id), self.rel_im_embeddings(r_id)
        b_size = len(h_id)
        if mode == 'entities':
            re_candidates = self.ent_re_embeddings.weight.data.view(1, self.ent_tot, self.dim)
            re_candidates = re_candidates.expand(b_size, self.ent_tot, self.dim)

            im_candidates = self.ent_im_embeddings.weight.data.view(1, self.ent_tot, self.dim)
            im_candidates = im_candidates.expand(b_size, self.ent_tot, self.dim)
        else:
            re_candidates = self.rel_re_embeddings.weight.data.view(1, self.rel_tot, self.dim)
            re_candidates = re_candidates.expand(b_size, self.ent_tot, self.dim)

            im_candidates = self.rel_im_embeddings.weight.data.view(1, self.rel_tot, self.dim)
            im_candidates = im_candidates.expand(b_size, self.rel_tot, self.dim)

        return (re_h, im_h), (re_r, im_r), (re_t, im_t), (re_candidates, im_candidates)

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        score = self.calc(h_re, h_im, t_re, t_im, r_re, r_im)
        return score

    def save(self):
        ent_re_embeddings = self.ent_re_embeddings.cpu().weight.data
        read.save_special_embeddings(self.out_folder, 'ent_re_embeddings', '', ent_re_embeddings, None)
        ent_im_embeddings = self.ent_im_embeddings.cpu().weight.data
        read.save_special_embeddings(self.out_folder, 'ent_im_embeddings', '', ent_im_embeddings, None)
        rel_re_embeddings = self.rel_re_embeddings.cpu().weight.data
        read.save_special_embeddings(self.out_folder, 'rel_re_embeddings', '', rel_re_embeddings, None)
        rel_im_embeddings = self.rel_im_embeddings.cpu().weight.data
        read.save_special_embeddings(self.out_folder, 'rel_im_embeddings', '', rel_im_embeddings, None)