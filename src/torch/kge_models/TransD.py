import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import empty
from torch.nn import Parameter
from tqdm import tqdm

from src.py.load import read
from src.py.util.util import to_var
from src.torch.kge_models.basic_model import BasicModel


class TransD(BasicModel):

    def __init__(self, args, kgs, dim_e=100, dim_r=100, p_norm=1, norm_flag=True, margin=None, epsilon=None):
        super(TransD, self).__init__(args, kgs)

        self.projected = False
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.norm_flag = norm_flag
        self.p_norm = 1

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_transfer = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_transfer = nn.Embedding(self.rel_tot, self.dim_r)
        self.projected_entities = Parameter(empty(size=(self.rel_tot,
                                                        self.ent_tot,
                                                        self.dim_r)),
                                            requires_grad=False)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_transfer.weight.data)
        nn.init.xavier_uniform_(self.rel_transfer.weight.data)

    def _resize(self, tensor, axis, size):
        shape = tensor.size()
        osize = shape[axis]
        if osize == size:
            return tensor
        if (osize > size):
            return torch.narrow(tensor, axis, 0, size)
        paddings = []
        for i in range(len(shape)):
            if i == axis:
                paddings = [0, size - osize] + paddings
            else:
                paddings = [0, 0] + paddings
        print(paddings)
        return F.pad(tensor, pad=paddings, mode="constant", value=0)

    def calc(self, h, r, t):
        '''h = F.normalize(h, 2, -1)
        r = F.normalize(r, 2, -1)
        t = F.normalize(t, 2, -1)'''
        score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1)
        return score

    def save(self):
        ent_embeds = self.ent_embeddings.cpu().weight.data
        rel_embeds = self.rel_embeddings.cpu().weight.data
        read.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=None)
        e_transfer = self.ent_transfer.cpu().weight.data
        r_transfer = self.rel_transfer.cpu().weight.data
        read.save_special_embeddings(self.out_folder, 'ent_transfer', 'rel_transfer', e_transfer, r_transfer)

    def _transfer(self, e, e_transfer, r_transfer):
        b_size = e.shape[0]

        scalar_product = (e * e_transfer).sum(dim=1)
        proj_e = (r_transfer * scalar_product.view(b_size, 1))
        return proj_e + e[:, :self.dim_r]

    def get_embeddings(self, hid, rid, tid, mode = 'entity'):
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
            candidates = self.rel_embeddings.weight.data.view(1, self.rel_tot, self.dim_r)
            candidates = candidates.expand(b_size, self.rel_tot, self.dim_r)
            return proj_h, r_embs, proj_t, candidates

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_transfer = self.ent_transfer(batch_h)
        t_transfer = self.ent_transfer(batch_t)
        r_transfer = self.rel_transfer(batch_r)
        h = self._transfer(h, h_transfer, r_transfer)
        t = self._transfer(t, t_transfer, r_transfer)
        score = self.calc(h, r, t).flatten()
        return score

    def evaluate_projections(self):
        """Link prediction evaluation helper function. Project all entities
        according to each relation. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        """
        if not self.projected:
            for i in tqdm(range(self.ent_tot), unit='entities', desc='Projecting entities'):
                rel_proj_vects = self.rel_transfer.weight.data
                ent = self.ent_embeddings.weight[i]
                ent_proj_vect = self.ent_transfer.weight[i]

                sc_prod = (ent_proj_vect * ent).sum(dim=0)
                proj_e = sc_prod * rel_proj_vects + ent[:self.dim_r].view(1, -1)
                self.projected_entities[:, i, :] = proj_e

                del proj_e
            self.projected = True
