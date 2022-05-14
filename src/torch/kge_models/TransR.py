import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, empty
from torch.cuda import empty_cache
from torch.nn import Parameter
from tqdm import tqdm

from .basic_model import BasicModel
from ...py.load import read
from ...py.util.util import to_var


class TransR(BasicModel):

    def __init__(self, kgs, args):
        super(TransR, self).__init__(args, kgs)
        self.dim_e = self.args.dim
        self.dim_r = self.args.dim
        self.projected = False
        self.norm_flag = self.args.ent_l2_norm
        self.p_norm = 1
        self.projected_entities = Parameter(empty(size=(self.rel_tot,
                                                        self.ent_tot,
                                                        self.dim_r)),
                                            requires_grad=False)
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.transfer_matrix = nn.Embedding(self.rel_tot, self.dim_e * self.dim_r)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.transfer_matrix.weight.data)

    def calc(self, h, t, r):
        h = F.normalize(h, 2, -1)
        # r = F.normalize(r, 2, -1)
        t = F.normalize(t, 2, -1)
        score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1)
        return score

    def transfer(self, e, r_transfer):
        proj_e = torch.matmul(e.view(-1, 1, self.dim_e), r_transfer)
        return proj_e.view(-1, self.dim_r)

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_transfer = self.transfer_matrix(batch_r).view(-1, self.dim_e, self.dim_r)
        h = self.transfer(h, r_transfer)
        t = self.transfer(t, r_transfer)
        score = self.calc(h, t, r).flatten()

    def get_embeddings(self, hid, rid, tid, mode='entity'):
        h = to_var(hid, self.device)
        r = to_var(rid, self.device)
        t = to_var(tid, self.device)
        self.evaluate_projections()
        r_embs = self.rel_embeddings(r)

        if mode == 'entity':
            proj_h = self.projected_entities[rid, hid].unsqueeze(1)  # shape: (b_size, 1, emb_dim)
            proj_t = self.projected_entities[rid, tid].unsqueeze(1)  # shape: (b_size, 1, emb_dim)
            candidates = self.projected_entities[rid]  # shape: (b_size, self.n_rel, self.emb_dim)
            return proj_h, r_embs, proj_t, candidates
        else:
            proj_h = self.projected_entities[:, hid].transpose(0, 1)  # shape: (b_size, n_rel, emb_dim)
            b_size = proj_h.shape[0]
            proj_t = self.projected_entities[:, tid].transpose(0, 1)  # shape: (b_size, n_rel, emb_dim)
            candidates = self.rel_embeddings.weight.data.view(1, self.rel_tot, self.dim_r)
            candidates = candidates.expand(b_size, self.rel_tot, self.dim_r)
            return proj_h, r_embs, proj_t, candidates

    def save(self):
        ent_embeds = self.ent_embeddings.cpu().weight.data
        rel_embeds = self.rel_embeddings.cpu().weight.data
        read.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=None)
        transfer_matrix = self.transfer_matrix.cpu().weight.data
        read.save_special_embeddings(self.out_folder, 'transfer_matrix', '', transfer_matrix, None)

    def evaluate_projections(self):
        """Link prediction evaluation helper function. Project all entities
        according to each relation. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        """
        if self.projected:
            return
        for i in tqdm(range(self.ent_tot), unit='entities', desc='Projecting entities'):
            '''if projection_matrices.is_cuda:
                empty_cache()'''
            projection_matrices = self.transfer_matrix.weight.data.view(self.rel_tot, self.dim_e, self.dim_r)
            mask = tensor([i], device=self.device).long()

            ent = self.ent_embeddings(mask)
            proj_ent = torch.matmul(ent.view(1, self.dim_e), projection_matrices)
            # if projection_matrices.is_cuda:
            #    empty_cache()
            # proj_ent = proj_ent.view(self.rel_tot, self.dim_r, 1)
            self.projected_entities[:, i, :] = proj_ent.view(self.rel_tot, self.dim_r)
            empty_cache()
            del proj_ent, projection_matrices
            # gc.collect()
            # if i % 100 == 0:
            # gc.collect()

        self.projected = True
