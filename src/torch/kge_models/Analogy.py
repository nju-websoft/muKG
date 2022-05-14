import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_model import BasicModel
from ...py.load import read
from ...py.util.util import to_var


class Analogy(BasicModel):

    def __init__(self, kgs, args):
        super(Analogy, self).__init__(args, kgs)
        self.dim = self.args.dim
        self.ent_re_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.ent_im_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_re_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.rel_im_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim*2)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim*2)
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)

    def calc(self, h_re, h_im, h, t_re, t_im, t, r_re, r_im, r):
        return -((h * r * t).sum(dim=1) +
                 (h_re * (r_re * t_re + r_im * t_im) + h_im * (r_re * t_im - r_im * t_re)).sum(dim=1))

    def get_score(self, h, r, t):
        sc_h, re_h, im_h = h[0], h[1], h[2]
        sc_t, re_t, im_t = t[0], t[1], t[2]
        sc_r, re_r, im_r = r[0], r[1], r[2]
        b_size = re_h.shape[0]

        if len(re_t.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_r.shape) == 2)
            # this is the tail completion case in link prediction
            return -((sc_h * sc_r).view(b_size, 1, self.dim * 2) * sc_t).sum(dim=2) \
                   - ((re_h * re_r - im_h * im_r).view(b_size, 1, self.dim) * re_t
                    + (re_h * im_r + im_h * re_r).view(b_size, 1, self.dim) * im_t).sum(dim=2)

        elif len(re_h.shape) == 3:
            assert (len(re_t.shape) == 2) & (len(re_r.shape) == 2)
            # this is the head completion case in link prediction
            return - (sc_h * (sc_r * sc_t).view(b_size, 1, self.dim * 2)).sum(dim=2) \
                   - (re_h * (re_r * re_t + im_r * im_t).view(b_size, 1, self.dim)
                    + im_h * (re_r * im_t - im_r * re_t).view(b_size, 1, self.dim)).sum(dim=2)

        elif len(re_r.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_t.shape) == 2)
            # this is the relation prediction case
            return - (sc_r * (sc_h * sc_t).view(b_size, 1, self.dim * 2)).sum(dim=2) \
                   - (re_r * (re_h * re_t + im_h * im_t).view(b_size, 1, self.dim)
                   + im_r * (re_h * im_t - im_h * re_t).view(b_size, 1, self.dim)).sum(dim=2)

    def get_embeddings(self, h_id, r_id, t_id, mode='entities'):
        h_id = to_var(h_id, self.device)
        r_id = to_var(r_id, self.device)
        t_id = to_var(t_id, self.device)
        b_size = h_id.shape[0]

        sc_h = self.ent_embeddings(h_id)
        re_h = self.ent_re_embeddings(h_id)
        im_h = self.ent_im_embeddings(h_id)

        sc_t = self.ent_embeddings(t_id)
        re_t = self.ent_re_embeddings(t_id)
        im_t = self.ent_im_embeddings(t_id)

        sc_r = self.rel_embeddings(r_id)
        re_r = self.rel_re_embeddings(r_id)
        im_r = self.rel_im_embeddings(r_id)

        if mode == 'entities':
            sc_candidates = self.ent_embeddings.weight.data
            sc_candidates = sc_candidates.view(1, self.ent_tot, -1)
            sc_candidates = sc_candidates.expand(b_size, self.ent_tot, self.dim * 2)

            re_candidates = self.ent_re_embeddings.weight.data
            re_candidates = re_candidates.view(1, self.ent_tot, self.dim)
            re_candidates = re_candidates.expand(b_size, self.ent_tot, self.dim)

            im_candidates = self.ent_im_embeddings.weight.data
            im_candidates = im_candidates.view(1, self.ent_tot, self.dim)
            im_candidates = im_candidates.expand(b_size, self.ent_tot, self.dim)

        else:
            sc_candidates = self.rel_embeddings.weight.data
            sc_candidates = sc_candidates.view(1, self.rel_tot, self.dim * 2)
            sc_candidates = sc_candidates.expand(b_size, self.rel_tot, self.dim * 2)

            re_candidates = self.rel_re_embeddings.weight.data
            re_candidates = re_candidates.view(1, self.rel_tot, self.dim)
            re_candidates = re_candidates.expand(b_size, self.rel_tot, self.dim)

            im_candidates = self.rel_im_embeddings.weight.data
            im_candidates = im_candidates.view(1, self.rel_tot, self.dim)
            im_candidates = im_candidates.expand(b_size, self.rel_tot, self.dim)

        return (sc_h, re_h, im_h), \
               (sc_r, re_r, im_r), \
               (sc_t, re_t, im_t), \
               (sc_candidates, re_candidates, im_candidates)

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        h = self.ent_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        t = self.ent_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        r = self.rel_embeddings(batch_r)
        score = self.calc(h_re, h_im, h, t_re, t_im, t, r_re, r_im, r)
        return score

    def save(self):
        ent_embeds = self.ent_embeddings.cpu().weight.data if self.ent_embeddings is not None else None
        rel_embeds = self.rel_embeddings.cpu().weight.data if self.rel_embeddings is not None else None
        read.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=None)
        read.save_special_embeddings(self.out_folder, 'ent_re_embeddings', '', self.ent_re_embeddings.cpu().weight.data, None)
        read.save_special_embeddings(self.out_folder, 'ent_im_embeddings', '', self.ent_im_embeddings.cpu().weight.data, None)
        read.save_special_embeddings(self.out_folder, 'rel_re_embeddings', '', self.rel_re_embeddings.cpu().weight.data, None)
        read.save_special_embeddings(self.out_folder, 'rel_im_embeddings', '', self.rel_im_embeddings.cpu().weight.data, None)
