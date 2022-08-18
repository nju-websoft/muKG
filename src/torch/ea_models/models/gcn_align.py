import gc
import math
import multiprocessing as mp
import random
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from src.py.util.util import to_var, to_tensor_cpu, to_tensor, early_stop, generate_out_folder
'''
Refactoring based on https://github.com/1049451037/GCN-Align
'''
_LAYER_UIDS = {}


# ******************************inits************************
def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    tmp = torch.Tensor(shape[0], shape[1])
    initial = nn.init.uniform_(tmp, -scale, scale)
    return tmp


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    t = torch.FloatTensor(shape[0], shape[1])
    tmp = nn.Parameter(t)
    nn.init.uniform_(tmp, -init_range, init_range)
    return tmp


def zeros(shape, name=None):
    """All zeros."""
    initial = nn.Parameter(torch.FloatTensor(shape[0]))
    nn.init.zeros_(initial)
    return initial

def ones(shape, name=None):
    """All zeros."""
    initial = nn.Parameter(torch.FloatTensor(shape[0], shape[1]))
    nn.init.ones_(initial)
    return initial

def trunc_normal(shape, name=None, normalize=True):
    tmp = torch.Tensor(shape[0] * shape[1])
    with torch.no_grad():
        size = shape
        tmp = tmp.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tmp.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tmp.data.mul_(1.0 / math.sqrt(shape[0])).add_(0)
    # initial = tf.Variable(tf.truncated_normal(shape, stddev=1.0 / math.sqrt(shape[0])))
    if not normalize:
        return tmp
    return F.normalize(tmp, 2, -1)


# *******************************layers**************************
def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    mask = ((torch.rand(x.values().size()) + keep_prob).floor()).type(torch.bool)
    rc = x.indices()[:, mask]
    val = x.values()[mask] * (1.0 / keep_prob)
    return torch.sparse.Tensor(rc, val)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    print(x)
    '''if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)'''
    res = torch.matmul(x, y)
    return res


def load_attr(ent_num, kgs):
    cnt = {}
    entity_attributes_dict = {**kgs.kg1.entity_attributes_dict, **kgs.kg2.entity_attributes_dict}
    for _, vs in entity_attributes_dict.items():
        for v in vs:
            if v not in cnt:
                cnt[v] = 1
            else:
                cnt[v] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    print(fre)
    attr2id = {}
    num = int(0.7 * len(cnt))
    for i in range(num):
        attr2id[fre[i][0]] = i
    attr = np.zeros((ent_num, num), dtype=np.float32)
    for ent, vs in entity_attributes_dict.items():
        for v in vs:
            if v in attr2id:
                attr[ent][attr2id[v]] = 1.0
    return attr


'''class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])'''


'''class Dense(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)'''


class GraphConvolution(nn.Module):
    """Graph convolution layer. (featureless=True and transform=False) is not supported for now."""

    def __init__(self, input_dim, output_dim, support, dropout=0.,
                 sparse_inputs=False, act=torch.relu, bias=False,
                 featureless=False, transform=True, **kwargs):
        super(GraphConvolution, self).__init__()

        '''if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.'''
        self.dropout = 0.
        self.act = act
        self.support = support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.transform = transform
        self.vars = {}
        # helper variable for sparse dropout
        self.num_features_nonzero = 0
        self.weight0 = None
        self.bias = self.zeros([output_dim])
        for i in range(len(self.support)):
            if input_dim == output_dim and not self.transform and not featureless:
                continue
            self.weight0 = self.glorot([input_dim, output_dim])
        if self.bias:
            self.vars['bias'] = self.zeros([output_dim])

        '''if self.logging:
            self._log_vars()'''

    def glorot(self, shape, name=None):
        """Glorot & Bengio (AISTATS 2010) init."""
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        t = torch.FloatTensor(shape[0], shape[1])
        tmp = nn.Parameter(t)
        nn.init.uniform_(tmp, -init_range, init_range)
        return tmp

    def zeros(self, shape, name=None):
        """All zeros."""
        initial = nn.Parameter(torch.FloatTensor(shape))
        nn.init.zeros_(initial)
        return initial

    def forward(self, data):
        x = data['features']

        # dropout
        if self.dropout:
            if self.sparse_inputs:
                x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
            else:
                x = torch.dropout(x, 1 - self.dropout, True)

        # convolve
        supports = list()
        output = None
        '''for i in range(len(self.support)):
            if 'weights_' + str(i) in self.vars:
                if not self.featureless:
                    pre_sup = torch.matmul(x, self.vars['weights_' + str(i)])
                else:
                    pre_sup = self.vars['weights_' + str(i)]
            else:
                pre_sup = x'''
        if self.weight0 is not None:
            if not self.featureless:
                pre_sup = torch.matmul(x, self.weight0)
            else:
                pre_sup = self.weight0
        else:
            pre_sup = x
        support = torch.matmul(data['support'].to(torch.float32), pre_sup)
        output = support
        supports.append(support)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)




def align_loss(outlayer, data, gamma, k):
    left = data['ILL0'].to(torch.int32)
    right = data['ILL1'].to(torch.int32)
    t = len(left)
    left_x = torch.index_select(outlayer, 0, left)
    right_x = torch.index_select(outlayer, 0, right)
    A = torch.sum(torch.abs(left_x - right_x), 1).unsqueeze(-1)
    neg_left = data['neg_left'].to(torch.int32)
    neg_right = data['neg_right'].to(torch.int32)
    neg_l_x = torch.index_select(outlayer, 0, neg_left)
    neg_r_x = torch.index_select(outlayer, 0, neg_right)
    B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1).unsqueeze(-1)
    C = - B.view(t, k)
    D = A + gamma
    L1 = torch.relu(C + D.view(t, 1))
    neg_left = data['neg2_left'].to(torch.int32)
    neg_right = data['neg2_right'].to(torch.int32)
    neg_l_x = torch.index_select(outlayer, 0, neg_left)
    neg_r_x = torch.index_select(outlayer, 0, neg_right)
    B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1).unsqueeze(-1)
    C = - B.view(t, k)
    L2 = torch.relu(C + D.view(t, 1))
    return (torch.sum(L1) + torch.sum(L2)) / (2.0 * k * t)


# ***************************models****************************************


class GCN_Align_Unit(nn.Module):
    def __init__(self, args, support, input_dim, output_dim, sparse_inputs=False, featureless=True, **kwargs):
        super(GCN_Align_Unit, self).__init__()
        self.outputs = None
        self.args = args
        self.layers = nn.ModuleList()
        # self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.support = support
        # self.ILL = ILL
        # self.layers = []
        self.activations = []
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.gru1 = GraphConvolution(input_dim=self.input_dim,
                                     output_dim=self.output_dim,
                                     support=self.support,
                                     act=torch.relu,
                                     dropout=False,
                                     featureless=self.featureless,
                                     sparse_inputs=self.sparse_inputs,
                                     transform=False
                                     )
        self.gru2 = GraphConvolution(input_dim=self.output_dim,
                                     output_dim=self.output_dim,
                                     support=self.support,
                                     act=lambda x: x,
                                     dropout=False,
                                     transform=False
                                     )
        print(list(self.gru1.parameters()))
        self.layers.append(self.gru1)
        self.layers.append(self.gru2)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.args.learning_rate)

    def loss(self, data):
        return align_loss(self.outputs, data, self.args.gamma, self.args.neg_triple_num)

    def _accuracy(self):
        pass

    def forward(self, data):
        """ Wrapper for _build() """
        # Build sequential layer model
        hidden = self.layers[0](data)
        data['features'] = hidden
        self.outputs = self.layers[1](data)
        '''for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]'''

        # Store model variables for easy access
        # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        # self.vars = {var.name: var for var in variables}

        # Build metrics
        return self.loss(data)
        # self._accuracy()

    def get_output(self):
        return self.outputs.cpu().detach().numpy()


class GCN_Utils:
    def __init__(self, args, kgs):
        self.args = args
        self.kgs = kgs

    @staticmethod
    def sparse_to_tuple(sparse_mx):
        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    @staticmethod
    def sparse_to_tensor(sparse_mx):
        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = []
            coords.append(mx.row)
            coords.append(mx.col)
            # coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return torch.sparse_coo_tensor(to_tensor_cpu(coords), values, size=shape)
            # return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    @staticmethod
    def normalize_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        adj_normalized = self.normalize_adj(adj + sp.eye(adj.shape[0]))
        return self.sparse_to_tensor(adj_normalized)

    @staticmethod
    def construct_feed_dict(features, support, placeholders):
        """Construct feed dictionary for GCN-Align."""
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
        return feed_dict

    def chebyshev_polynomials(self, adj, k):
        """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
        print("Calculating Chebyshev polynomials up to order {}...".format(k))

        adj_normalized = self.normalize_adj(adj)
        laplacian = sp.eye(adj.shape[0]) - adj_normalized
        largest_eigval, _ = eigsh(laplacian, 1, which='LM')
        scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

        t_k = list()
        t_k.append(sp.eye(adj.shape[0]))
        t_k.append(scaled_laplacian)

        def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
            s_lap = sp.csr_matrix(scaled_lap, copy=True)
            return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

        for i in range(2, k + 1):
            t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

        return self.sparse_to_tuple(t_k)

    @staticmethod
    def func(triples):
        head = {}
        cnt = {}
        for tri in triples:
            if tri[1] not in cnt:
                cnt[tri[1]] = 1
                head[tri[1]] = {tri[0]}
            else:
                cnt[tri[1]] += 1
                head[tri[1]].add(tri[0])
        r2f = {}
        for r in cnt:
            r2f[r] = len(head[r]) / cnt[r]
        return r2f

    @staticmethod
    def ifunc(triples):
        tail = {}
        cnt = {}
        for tri in triples:
            if tri[1] not in cnt:
                cnt[tri[1]] = 1
                tail[tri[1]] = {tri[2]}
            else:
                cnt[tri[1]] += 1
                tail[tri[1]].add(tri[2])
        r2if = {}
        for r in cnt:
            r2if[r] = len(tail[r]) / cnt[r]
        return r2if

    def get_weighted_adj(self, e, KG):
        r2f = self.func(KG)
        r2if = self.ifunc(KG)
        M = {}
        for tri in KG:
            if tri[0] == tri[2]:
                continue
            if (tri[0], tri[2]) not in M:
                M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
            else:
                M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
            if (tri[2], tri[0]) not in M:
                M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
            else:
                M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
        row = []
        col = []
        data = []
        for key in M:
            row.append(key[1])
            col.append(key[0])
            data.append(M[key])
        indice = []
        indice.append(row)
        indice.append(col)
        # return torch.sparse_coo_tensor(to_tensor(indice), data, size=[e, e])
        return sp.coo_matrix((data, (row, col)), shape=(e, e))

    def get_ae_input(self, attr):
        return self.sparse_to_tuple(sp.coo_matrix(attr))

    def load_data(self, attr):
        ae_input = self.get_ae_input(attr)
        triples = self.kgs.kg1.relation_triples_list + self.kgs.kg2.relation_triples_list
        adj = self.get_weighted_adj(self.kgs.entities_num, triples)
        train = np.array(self.kgs.train_links)
        return adj, ae_input, train



