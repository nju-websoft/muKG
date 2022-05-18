import time
import tensorflow as tf

from src.py.base.initializers import init_embeddings
from src.py.base.losses import get_loss_func_tf, get_loss_func_tfv2

tf.compat.v1.enable_eager_execution()
from src.tf.kge_models.basic_model import BasicModel


class TransE(BasicModel):

    def __init__(self, kgs, args):
        super(TransE, self).__init__(args, kgs)
        self.dim = self.args.dim
        self.margin = self.args.margin
        # self.epsilon = epsilon
        self.p_norm = 'L1'

        self.ent_embeddings = init_embeddings([self.ent_tot, self.dim], 'ent_embeddings',
                                              'xavier', False, dtype=tf.float32)
        # 初始化关系翻译向量空间
        self.rel_embeddings = init_embeddings([self.rel_tot, self.dim], 'rel_embeddings',
                                              'xavier', False, dtype=tf.float32)

    def calc(self, h, r, t):
        if self.p_norm == 'L1':
            score = tf.math.reduce_sum(tf.math.abs(h + r - t), -1)
        else:
            score = tf.math.reduce_sum((h + r - t) ** 2, -1)
        # neg = tf.math.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
        # return tf.math.reduce_sum(tf.math.maximum(pos - neg + self.margin, 0))
        return score

    def get_embeddings(self, h, r, t, mode='entity'):
        r_embs = tf.expand_dims(tf.nn.embedding_lookup(self.rel_embeddings, r), 1)
        proj_h = tf.expand_dims(tf.nn.embedding_lookup(self.ent_embeddings, h), 1)  # shape: (b_size, 1, emb_dim)
        b_size = proj_h.get_shape().as_list()[0]
        proj_t = tf.expand_dims(tf.nn.embedding_lookup(self.ent_embeddings, t), 1)  # shape: (b_size, 1, emb_dim)
        if mode == 'entity':
            candidates = tf.reshape(self.ent_embeddings, [1, self.ent_tot, self.dim])
            candidates = tf.tile(candidates, [b_size, 1, 1])
            return proj_h, r_embs, proj_t, candidates
        else:
            candidates = tf.reshape(self.rel_embeddings, [1, self.rel_tot, self.dim])
            candidates = tf.tile(candidates, [b_size, 1, 1])
            return proj_h, r_embs, proj_t, candidates

    def call(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = tf.nn.embedding_lookup(self.ent_embeddings, batch_h)
        r = tf.nn.embedding_lookup(self.rel_embeddings, batch_r)
        t = tf.nn.embedding_lookup(self.ent_embeddings, batch_t)
        score = self.calc(h, r, t)
        #self.batch_size = int(len(batch_h) / (self.args.neg_triple_num + 1))
        #po_score = self.get_pos_score(score)
        #ne_score = self.get_neg_score(score)
        #score = get_loss_func_tfv2(po_score, ne_score, self.args)
        # score = tf.math.reduce_sum(tf.math.maximum(po_score - ne_score + self.margin, 0))
        return score
