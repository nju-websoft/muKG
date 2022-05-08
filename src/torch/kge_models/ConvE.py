import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.py.util.util import to_var
from src.torch.kge_models.basic_model import BasicModel


class ConvE(BasicModel):
    def __init__(self, args, kgs):
        super(ConvE, self).__init__(args, kgs)
        self.ent_embeddings = torch.nn.Embedding(self.ent_tot, self.args.dim, padding_idx=0)
        self.rel_embeddings = torch.nn.Embedding(self.rel_tot, self.args.dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(self.args.input_drop)
        self.hidden_drop = torch.nn.Dropout(self.args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(self.args.feat_drop)
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = self.args.embedding_shape1
        self.emb_dim2 = self.args.dim // self.emb_dim1
        if self.args.init == 'xavier':
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            nn.init.xavier_normal_(self.ent_embeddings.weight.data)
            nn.init.xavier_normal_(self.rel_embeddings.weight.data)
        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), (1, 1), 0, bias=self.args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.args.dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.ent_tot)))
        self.fc = torch.nn.Linear(self.args.hidden_size, self.args.dim)

    def cal(self, e1_embedded, rel_embedded):
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.ent_embeddings.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        return x

    def forward(self, data):
        batch_h = data['batch_h']
        batch_r = data['batch_r']
        batch_t = data['batch_t']
        t = self.get_batch(batch_h.shape[0], batch_t)
        e1_embedded = self.ent_embeddings(batch_h).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.rel_embeddings(batch_r).view(-1, 1, self.emb_dim1, self.emb_dim2)
        x = self.cal(e1_embedded, rel_embedded)
        pred = torch.sigmoid(x)
        return self.loss(pred, t)

    def get_batch(self, batch_size, batch_t):
        # targets = np.zeros((batch_size, self.ent_tot))
        # for idx, t in enumerate(batch_t):
        #    targets[idx, t] = 1.
        # targets = torch.FloatTensor(targets).to(self.device)

        targets = torch.zeros(batch_size, self.ent_tot).scatter_(1, batch_t.cpu().view(-1, 1).type(torch.int64), 1).to(
            self.device)
        return targets

    def get_embeddings(self, h_idx, r_idx, t_idx, mode='entities'):
        h_idx = to_var(h_idx, self.device)
        r_idx = to_var(r_idx, self.device)
        h_emb = self.ent_embeddings(h_idx).view(-1, 1, self.emb_dim1, self.emb_dim2)
        r_emb = self.rel_embeddings(r_idx).view(-1, 1, self.emb_dim1, self.emb_dim2)
        return h_emb, r_emb, None, None

    def get_score(self, h, r, t):
        return -self.cal(h, r)