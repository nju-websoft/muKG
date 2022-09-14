import gc
import math
import time
import numpy as np
import torch
import torch.nn as nn
import scipy
import pandas as pd
import string

from torch.autograd import Variable

from src.py.base.optimizers import get_optimizer_torch
from src.py.util.util import to_tensor_cpu, to_tensor
from src.torch.ea_models.models.gcn_align import glorot
from src.torch.kge_models.basic_model import align_model_trainer


def rfunc(triple_list, ent_num, rel_num):
    head = dict()
    tail = dict()
    rel_count = dict()
    r_mat_ind = list()
    r_mat_ind_x = list()
    r_mat_ind_y = list()
    r_mat_val = list()
    head_r = np.zeros((ent_num, rel_num))
    tail_r = np.zeros((ent_num, rel_num))
    for triple in triple_list:
        head_r[triple[0]][triple[1]] = 1
        tail_r[triple[2]][triple[1]] = 1
        r_mat_ind_x.append(triple[0])
        r_mat_ind_y.append(triple[2])
        # r_mat_ind.append([triple[0], triple[2]])
        r_mat_val.append(triple[1])
        if triple[1] not in rel_count:
            rel_count[triple[1]] = 1
            head[triple[1]] = set()
            tail[triple[1]] = set()
            head[triple[1]].add(triple[0])
            tail[triple[1]].add(triple[2])
        else:
            rel_count[triple[1]] += 1
            head[triple[1]].add(triple[0])
            tail[triple[1]].add(triple[2])
    r_mat_ind.append(r_mat_ind_x)
    r_mat_ind.append(r_mat_ind_y)
    del r_mat_ind_y
    del r_mat_ind_x
    gc.collect()
    r_mat = torch.sparse_coo_tensor(indices=to_tensor_cpu(r_mat_ind), values=r_mat_val, size=[ent_num, ent_num])

    return head, tail, head_r, tail_r, r_mat, to_tensor_cpu(r_mat_ind), to_tensor_cpu(r_mat_val), [ent_num, ent_num]


def get_mat(triple_list, ent_num):
    degree = [1] * ent_num
    pos = dict()
    for triple in triple_list:
        if triple[0] != triple[1]:
            degree[triple[0]] += 1
            degree[triple[1]] += 1
        if triple[0] == triple[2]:
            continue
        if (triple[0], triple[2]) not in pos:
            pos[(triple[0], triple[2])] = 1
            pos[(triple[2], triple[0])] = 1

    for i in range(ent_num):
        pos[(i, i)] = 1
    return pos, degree


def get_sparse_tensor(triple_list, ent_num):
    pos, degree = get_mat(triple_list, ent_num)
    ind = []
    ind_x = []
    ind_y = []
    val = []
    M_arr = np.zeros((ent_num, ent_num))
    for fir, sec in pos:
        ind_x.append(sec)
        ind_y.append(fir)
        # ind.append((sec, fir))
        val.append(pos[(fir, sec)] / math.sqrt(degree[fir]) / math.sqrt(degree[sec]))
        M_arr[fir][sec] = 1.0
    ind.append(ind_x)
    ind.append(ind_y)
    del ind_y
    del ind_x
    gc.collect()
    pos = torch.sparse_coo_tensor(indices=to_tensor_cpu(ind), values=val, size=[ent_num, ent_num])

    return pos, M_arr


def get_neg(ILL, output_layer, k):
    neg = []
    t = len(ILL)
    if output_layer is None:
        output_layer = np.load("D:/OPENEA-pytorch/a.npy")

    ILL_vec = np.array([output_layer[e1] for e1 in ILL])
    KG_vec = np.array(output_layer)
    sim = scipy.spatial.distance.cdist(ILL_vec, KG_vec, metric='cityblock')
    for i in range(t):
        rank = sim[i, :].argsort()
        neg.append(rank[0:k])

    neg = np.array(neg)
    neg = neg.reshape((t * k,))
    return neg


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


class Layer(nn.Module):
    def __init__(self, args, kg, embedding):
        super(Layer, self).__init__()
        self.drop = None
        self.kernel_gate = None
        self.tail_l = None
        self.head_l = None
        self.r_mat = None
        self.tail_r = None
        self.tail = None
        self.head = None
        self.head_r = None
        self.M_arr = None
        self.M = None
        self.conv7 = None
        self.pretrianed_embedding = None
        self.conv6 = None
        self.conv5 = None
        self.conv4 = None
        self.conv3 = None
        self.conv2 = None
        self.conv1 = None
        self.w1 = None
        self.primal_X_0 = None
        self.bias_gate = None
        self.w0 = None
        self.adj = None
        self.r_mat_indice = None
        self.r_mat_value = None
        self.r_mat_shape = None
        self.output_layer = None
        self.dim = args.dim
        self.dropout = args.dropout
        self.act_func = torch.relu
        self.gamma = args.gamma
        if args.is_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.ILL = to_tensor(np.array(kg.train_links), self.device)
        self.k = args.neg_triple_num
        self.alpha = args.alpha
        self.beta = args.beta
        self.triple_list = kg.kg1.relation_triples_list + kg.kg2.relation_triples_list
        self.rel_num = kg.relations_num
        self.ent_num = kg.entities_num

    def init(self):
        self.drop = nn.Dropout(self.dropout)
        self.w0 = self.ones((1, self.dim))
        self.bias_gate = self.zeros([self.dim])
        self.w1 = self.glorot((self.dim, self.dim))
        self.primal_X_0 = self.glorot((self.ent_num, self.dim))
        self.conv1 = nn.Conv1d(600, 1, (1,))
        self.conv2 = nn.Conv1d(600, self.dim, (1,))
        self.conv3 = nn.Conv1d(self.dim, 1, (1,))
        self.conv4 = nn.Conv1d(self.dim, 1, (1,))
        self.conv5 = nn.Conv1d(600, self.dim, (1,), bias=False)
        self.conv6 = nn.Conv1d(self.dim, 1, (1,))
        self.conv7 = nn.Conv1d(self.dim, 1, (1,))
        # self.pretrianed_embedding = embedding
        self.M, self.M_arr = get_sparse_tensor(self.triple_list, self.ent_num)
        self.head, self.tail, self.head_r, self.tail_r, self.r_mat, self.r_mat_indice, self.r_mat_value, self.r_mat_shape = rfunc(
            self.triple_list, self.ent_num, self.rel_num)
        self.r_mat = self.r_mat.to(self.device)
        self.r_mat_value = self.r_mat_value.to(self.device)
        self.r_mat_indice = self.r_mat_indice.to(self.device)
        self.M = self.M.to(self.device)
        self.head_l = to_tensor(self.head_r, self.device).T
        self.tail_l = to_tensor(self.tail_r, self.device).T
        self.kernel_gate = glorot([self.dim, self.dim])

    def glorot(self, shape, name=None):
        """Glorot & Bengio (AISTATS 2010) init."""
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        t = torch.FloatTensor(shape[0], shape[1])
        tmp = nn.Parameter(t)
        nn.init.uniform_(tmp, -init_range, init_range)
        return tmp

    def zeros(self, shape, name=None):
        """All zeros."""
        initial = nn.Parameter(torch.FloatTensor(shape[0]))
        nn.init.zeros_(initial)
        return initial

    def ones(self, shape, name=None):
        """All zeros."""
        initial = nn.Parameter(torch.FloatTensor(shape[0], shape[1]))
        nn.init.ones_(initial)
        return initial

    def add_diag_layer(self, inlayer, init=ones):
        inlayer = self.drop(inlayer)
        # w0 = init([1, self.dim])
        # b = self.w0.repeat(inlayer.shape[0], 1)
        # c = self.w0
        tosum = torch.matmul(self.M, torch.multiply(inlayer, self.w0))
        if self.act_func is None:
            return tosum
        else:
            return torch.relu(tosum)

    def add_full_layer(self, inlayer, init=glorot):
        inlayer = self.drop(inlayer)
        # w0 = init([self.dim, self.dim])
        tosum = torch.matmul(self.M, torch.matmul(inlayer, self.w1))
        if self.act_func is None:
            return tosum
        else:
            return self.act_func(tosum)

    def add_sparse_att_layer(self, inlayer, dual_layer):
        a = dual_layer.permute(1, 0).to(torch.float32)
        dual_transform = self.conv1(torch.unsqueeze(a, 0)).permute(0, 2, 1).view(-1, 1)
        logits = torch.index_select(dual_transform, 0, self.r_mat_value).squeeze()
        lrelu = torch.sparse_coo_tensor(indices=self.r_mat_indice,
                                        values=torch.relu(logits),
                                        size=self.r_mat_shape)
        coefs = torch.sparse.softmax(lrelu, 1)
        vals = torch.matmul(coefs.to_dense(), inlayer)
        if self.act_func is None:
            return vals
        else:
            return self.act_func(vals)

    def add_dual_att_layer(self, inlayer, inlayer2, adj):
        a = inlayer2.permute(1, 0).to(torch.float32)
        in_fts = self.conv2(torch.unsqueeze(a, 0))
        f_1 = self.conv3(in_fts).permute(0, 2, 1).view(-1, 1)
        f_2 = self.conv4(in_fts).permute(0, 2, 1).view(-1, 1)
        logits = f_1 + f_2.T
        adj_tensor = adj.detach()
        bias_mat = -1e9 * (~(adj > 0))
        logits = torch.multiply(adj_tensor, logits)
        coefs = torch.softmax(torch.relu(logits) + bias_mat, -1)

        vals = torch.matmul(coefs, inlayer)
        if self.act_func is None:
            return vals
        else:
            return self.act_func(vals)

    def add_self_att_layer(self, inlayer, adj):
        tmp = torch.unsqueeze(inlayer, 0).to(torch.float32)
        in_fts = self.conv5(tmp.permute(0, 2, 1))
        f_1 = self.conv6(in_fts).permute(0, 2, 1).view(-1, 1)
        f_2 = self.conv7(in_fts).permute(0, 2, 1).view(-1, 1)
        logits = f_1 + f_2.T
        adj_tensor = adj.detach()
        logits = torch.multiply(adj_tensor, logits)
        bias_mat = -1e9 * (~(adj > 0))
        coefs = torch.softmax(torch.relu(logits) + bias_mat, -1)

        vals = torch.matmul(coefs, inlayer)
        if self.act_func is None:
            return vals
        else:
            return self.act_func(vals)

    def highway(self, layer1, layer2):

        transform_gate = torch.matmul(layer1, self.kernel_gate) + self.bias_gate
        transform_gate = torch.sigmoid(transform_gate)
        carry_gate = 1.0 - transform_gate
        return transform_gate * layer2 + carry_gate * layer1

    def compute_r(self, inlayer):
        L = torch.matmul(self.head_l.to(torch.float32), inlayer.to(torch.float32)) / \
            torch.unsqueeze(torch.sum(self.head_l, dim=-1), -1)
        R = torch.matmul(self.tail_l.to(torch.float32), inlayer.to(torch.float32)) / \
            torch.unsqueeze(torch.sum(self.tail_l, dim=-1), -1)
        r_embeddings = torch.concat([L, R], dim=-1)
        return r_embeddings

    def get_dual_input(self, inlayer):
        dual_X = self.compute_r(inlayer)
        count_r = len(self.head)
        dual_A = np.zeros((count_r, count_r))
        for i in range(count_r):
            for j in range(count_r):
                a_h = len(self.head[i] & self.head[j]) / len(self.head[i] | self.head[j])
                a_t = len(self.tail[i] & self.tail[j]) / len(self.tail[i] | self.tail[j])
                dual_A[i][j] = a_h + a_t
        return dual_X, to_tensor(dual_A, self.device)

    # ******************************get_input_layer is used to initialize embeddings**********
    def get_input_layer(self):
        ent_embeddings = glorot((self.ent_num, self.dim))
        return ent_embeddings
        # input_embeddings = tf.random_uniform([self.ent_num, self.dim], minval=-1, maxval=1)
        # ent_embeddings = tf.Variable(input_embeddings)
        # return tf.nn.l2_normalize(ent_embeddings, 1)

    def get_pretrained_input(self, embedding):
        embedding = embedding.to(torch.float32)
        ent_embeddings = Variable(embedding)
        return ent_embeddings
        # return tf.nn.l2_normalize(ent_embeddings, 1)

    def get_loss(self, outlayer, data):
        left = self.ILL[:, 0]
        right = self.ILL[:, 1]
        t = len(self.ILL)
        left_x = torch.index_select(outlayer, 0, left)
        right_x = torch.index_select(outlayer, 0, right)
        A = torch.sum(torch.abs(left_x - right_x), 1)
        neg_left = data['neg_left'].to(torch.int32)
        neg_right = data['neg_right'].to(torch.int32)
        neg_l_x = torch.index_select(outlayer, 0, neg_left)
        neg_r_x = torch.index_select(outlayer, 0, neg_right)
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
        C = - B.view(t, self.k)
        D = A + self.gamma
        L1 = torch.relu(C + D.view(t, 1))
        neg_left = data['neg2_left'].to(torch.int32)
        neg_right = data['neg2_right'].to(torch.int32)
        neg_l_x = torch.index_select(outlayer, 0, neg_left)
        neg_r_x = torch.index_select(outlayer, 0, neg_right)
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
        C = - B.view(t, self.k)
        L2 = torch.relu(torch.add(C, D.view(t, 1)))
        return (torch.sum(L1) + torch.sum(L2)) / (2.0 * self.k * t)

    def build(self, data):
        # tf.reset_default_graph()
        dual_X_1, dual_A_1 = self.get_dual_input(self.primal_X_0)
        dual_H_1 = self.add_self_att_layer(dual_X_1, dual_A_1)
        primal_H_1 = self.add_sparse_att_layer(self.primal_X_0, dual_H_1)
        primal_X_1 = self.primal_X_0 + self.alpha * primal_H_1

        dual_X_2, dual_A_2 = self.get_dual_input(primal_X_1)
        dual_H_2 = self.add_dual_att_layer(dual_H_1, dual_X_2, dual_A_2)
        primal_H_2 = self.add_sparse_att_layer(primal_X_1, dual_H_2)
        primal_X_2 = self.primal_X_0 + self.beta * primal_H_2

        gcn_layer_1 = self.add_diag_layer(primal_X_2)
        gcn_layer_1 = self.highway(primal_X_2, gcn_layer_1)
        gcn_layer_2 = self.add_diag_layer(gcn_layer_1, )
        self.output_layer = self.highway(gcn_layer_1, gcn_layer_2)
        loss = self.get_loss(self.output_layer, data)
        return loss

    def get_output(self):
        if self.output_layer is None:
            self.output_layer = torch.randn(self.ent_num, self.dim)
        return self.output_layer.cpu().detach().numpy()

