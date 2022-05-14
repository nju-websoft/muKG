import math

import numpy as np
import torch
from torch.nn.init import xavier_normal_

from src.py.util.util import to_var
from src.torch.kge_models.basic_model import BasicModel


class TuckER(BasicModel):
    def __init__(self, args, kgs):
        super(TuckER, self).__init__(args, kgs)
        self.dim_e = self.args.dim_e
        self.dim_r = self.args.dim_r
        self.ent_embeddings = torch.nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = torch.nn.Embedding(self.rel_tot, self.dim_r)
        xavier_normal_(self.ent_embeddings.weight.data)
        xavier_normal_(self.rel_embeddings.weight.data)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.dim_r, self.dim_e, self.dim_e)),
                                                 dtype=torch.float, requires_grad=True))

        self.input_dropout = torch.nn.Dropout(self.args.input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(self.args.hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(self.args.hidden_dropout2)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(self.dim_e)
        self.bn1 = torch.nn.BatchNorm1d(self.dim_e)

    def cal(self, e1, r):
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.ent_embeddings.weight.transpose(1, 0))
        return x

    def forward(self, data):
        batch_h = data['batch_h']
        batch_r = data['batch_r']
        batch_t = data['batch_t']
        t = self.get_batch(batch_h.shape[0], batch_t)
        e1 = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)
        pred = torch.sigmoid(self.cal(e1, r))
        # t = ((1.0-0.1)*t) + (1.0/t.size(1))
        return self.loss(pred, t)

    def get_embeddings(self, h_idx, r_idx, t_idx, mode='entities'):
        h_idx = to_var(h_idx, self.device)
        r_idx = to_var(r_idx, self.device)
        h_emb = self.ent_embeddings(h_idx)
        r_emb = self.rel_embeddings(r_idx)
        return h_emb, r_emb, None, None

    def get_batch(self, batch_size, batch_t):
        # targets = np.zeros((batch_size, self.ent_tot))
        # for idx, t in enumerate(batch_t):
        #    targets[idx, t] = 1.
        # targets = torch.FloatTensor(targets).to(self.device)

        targets = torch.zeros(batch_size, self.ent_tot).scatter_(1, batch_t.cpu().view(-1, 1).type(torch.int64), 1).to(
            self.device)
        return targets

    def get_score(self, h, r, t):
        return -self.cal(h, r)
