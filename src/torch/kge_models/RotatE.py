import torch
import torch.autograd as autograd
import torch.nn as nn
import os
import numpy as np
from src.py.util.util import to_var
from src.torch.kge_models.basic_model import BasicModel


class RotatE(BasicModel):

    def __init__(self, args, kgs):
        super(RotatE, self).__init__(args, kgs)

        self.margin = self.args.margin
        self.epsilon = 2
        self.dim = self.args.dim
        self.dim_e = self.dim * 2
        self.dim_r = self.dim
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False
        assert self.args.init == 'uniform'
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim]),
            requires_grad=False
        )

        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )

        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim]),
            requires_grad=False
        )

        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )

        self.margin = nn.Parameter(torch.Tensor([self.margin]))
        self.margin.requires_grad = False

    def _calc(self, h, t, r):
        pi = self.pi_const

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        """
        re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)
        """
        re_head = re_head.view(re_relation.shape[0], 1, re_head.shape[-1])
        re_tail = re_tail.view(re_relation.shape[0], 1, re_tail.shape[-1])
        im_head = im_head.view(re_relation.shape[0], 1, im_head.shape[-1])
        im_tail = im_tail.view(re_relation.shape[0], 1, im_tail.shape[-1])
        im_relation = im_relation.view(re_relation.shape[0], 1, im_relation.shape[-1])
        re_relation = re_relation.view(re_relation.shape[0], 1, re_relation.shape[-1])

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head
        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        return score.squeeze()

    def generate_rank(self, h, t, re_relation, im_relation):

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)
        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head
        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        return score.squeeze()

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h, t, r) - self.margin
        return score

    def get_embeddings(self, h_idx, r_idx, t_idx, mode='entities'):
        h_idx = to_var(h_idx, self.device)
        r_idx = to_var(r_idx, self.device)
        t_idx = to_var(t_idx, self.device)
        b_size = h_idx.shape[0]
        h_emb = self.ent_embeddings(h_idx)
        t_emb = self.ent_embeddings(t_idx)
        r_emb = self.rel_embeddings(r_idx)
        if mode == 'entities':
            candidates = self.ent_embeddings.weight.data.view(1, self.ent_tot, self.dim_e)
            candidates = candidates.expand(b_size, self.ent_tot, self.dim_e)
        else:
            candidates = self.rel_embeddings.weight.data.view(1, self.rel_tot, self.dim_r)
            candidates = candidates.expand(b_size, self.rel_tot, self.dim_r)
        return h_emb, r_emb, t_emb, candidates

    def get_score(self, h, r, t):
        b_size = h.shape[0]
        phase_relation = r / (self.rel_embedding_range.item() / self.pi_const)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        if len(t.shape) == 3:
            assert (len(h.shape) == 2) & (len(r.shape) == 2)
            # this is the tail completion case in link prediction
            return self.generate_rank(h.view(-1, 1, self.dim_e), t,
                                      re_relation.view(-1, 1, self.dim_r),
                                      im_relation.view(-1, 1, self.dim_r))
        elif len(h.shape) == 3:
            assert (len(t.shape) == 2) & (len(r.shape) == 2)
            # this is the head completion case in link prediction
            return self.generate_rank(h, t.view(-1, 1, self.dim_e),
                                      re_relation.view(-1, 1, self.dim_r),
                                      im_relation.view(-1, 1, self.dim_r))
        elif len(r.shape) == 3:
            assert (len(h.shape) == 2) & (len(t.shape) == 2)
            return self.generate_rank(h.view(-1, 1, self.dim_e),
                                      t.view(-1, 1, self.dim_e), re_relation, im_relation)

    def load_embeddings(self):
        """
        This function we used for link prediction, firstly we load embeddings,
        the the evaluation class can simply pass the h, r, t ids to this function,
        the model returns the score.
        """
        dir = self.out_folder.split("/")
        new_dir = ""
        print(dir)
        for i in range(len(dir) - 1):
            new_dir += (dir[i] + "/")
        exist_file = os.listdir(new_dir)
        new_dir = new_dir + "/"
        ent_embeds = np.load(new_dir + "ent_embeds.npy")
        rel_embeds = np.load(new_dir + "rel_embeds.npy")
        self.ent_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(ent_embeds))
        self.rel_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(rel_embeds))
