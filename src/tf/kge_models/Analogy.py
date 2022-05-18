import math

import tensorflow as tf

from ...py.base.losses import get_loss_func_tfv2

tf.compat.v1.enable_eager_execution()
from .basic_model import BasicModel
#from ...py.base.initializers import init_embeddings_v2
from ...py.load import read
from ...py.util.util import to_var


class Analogy(BasicModel):

    def __init__(self, kgs, args, dim=100):
        super(Analogy, self).__init__(args, kgs)
        self.dim = dim
        self.ent_re_embeddings = tf.keras.layers.Embedding(
            input_dim=self.ent_tot, output_dim=self.dim, name="ent_re_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )

        self.ent_im_embeddings = tf.keras.layers.Embedding(
            input_dim=self.ent_tot, output_dim=self.dim, name="ent_im_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )

        self.rel_re_embeddings = tf.keras.layers.Embedding(
            input_dim=self.rel_tot, output_dim=self.dim, name="rel_re_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )
        self.rel_im_embeddings = tf.keras.layers.Embedding(
            input_dim=self.rel_tot, output_dim=self.dim, name="rel_im_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )

        self.ent_embeddings = tf.keras.layers.Embedding(
            input_dim=self.ent_tot, output_dim=self.dim, name="ent_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )

        self.rel_embeddings = tf.keras.layers.Embedding(
            input_dim=self.rel_tot, output_dim=self.dim, name="rel_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )

        """
        self.ent_re_embeddings = init_embeddings_v2([self.ent_tot, self.dim], 'ent_re_embeddings',
                                          self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
        self.ent_im_embeddings = init_embeddings_v2([self.ent_tot, self.dim], 'ent_im_embeddings',
                                                 self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
        self.rel_re_embeddings = init_embeddings_v2([self.rel_tot, self.dim], 'rel_re_embeddings',
                                                 self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
        #std = 1.0 / math.sqrt(100)
        #self.rel_im_embeddings = tf.Variable(initial_value=tf.random.truncated_normal(shape=[self.rel_tot, self.dim], stddev=std))
        self.rel_im_embeddings = init_embeddings_v2([self.rel_tot, self.dim], 'rel_im_embeddings',
                                                self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
        self.ent_embeddings = init_embeddings_v2([self.ent_tot, self.dim*2], 'ent_embeddings',
                                                 self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
        self.rel_embeddings = init_embeddings_v2([self.rel_tot, self.dim*2], 'rel_embeddings',
                                              self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
        """

    def calc(self, h_re, h_im, h, t_re, t_im, t, r_re, r_im, r):
        return -(tf.reduce_sum(h * r * t, 1)+
                 tf.reduce_sum(h_re * (r_re * t_re + r_im * t_im) + h_im * (r_re * t_im - r_im * t_re), 1))

    def get_score(self, h, r, t):
        sc_h, re_h, im_h = h[0], h[1], h[2]
        sc_t, re_t, im_t = t[0], t[1], t[2]
        sc_r, re_r, im_r = r[0], r[1], r[2]
        b_size = re_h.shape[0]

        if len(re_t.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_r.shape) == 2)
            # this is the tail completion case in link prediction
            return -tf.reduce_sum(tf.reshape(sc_h * sc_r, [b_size, 1, self.dim * 2]) * sc_t, 2) \
                   - tf.reduce_sum(tf.reshape(re_h * re_r - im_h * im_r, [b_size, 1, self.dim]) * re_t
                    + tf.reshape(re_h * im_r + im_h * re_r, [b_size, 1, self.dim]) * im_t, 2)

        elif len(re_h.shape) == 3:
            assert (len(re_t.shape) == 2) & (len(re_r.shape) == 2)
            # this is the head completion case in link prediction
            return -tf.reduce_sum(tf.reshape(sc_h * (sc_r * sc_t), [b_size, 1, self.dim * 2]), 2) \
                   - tf.reduce_sum(tf.reshape(re_h * (re_r * re_t + im_r * im_t), [b_size, 1, self.dim])
                    + tf.reshape(im_h * (re_r * im_t - im_r * re_t), [b_size, 1, self.dim]), 2)

        elif len(re_r.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_t.shape) == 2)
            # this is the relation prediction case
            return -tf.reduce_sum(tf.reshape(sc_r * (sc_h * sc_t), [b_size, 1, self.dim * 2]), 2) \
                   -tf.reduce_sum(tf.reshape(re_r * (re_h * re_t + im_h * im_t), [b_size, 1, self.dim])
                   + tf.reshape(im_r * (re_h * im_t - im_h * re_t), [b_size, 1, self.dim]), 2)

    def get_embeddings(self, h_id, r_id, t_id, mode='entities'):
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
            sc_candidates = self.ent_embeddings
            sc_candidates = tf.reshape(sc_candidates, [1, self.ent_tot, -1])
            sc_candidates = tf.tile(sc_candidates, [b_size, 1, 1])

            re_candidates = self.ent_re_embeddings
            re_candidates = tf.reshape(re_candidates, [1, self.ent_tot, -1])
            re_candidates = tf.tile(re_candidates, [b_size, 1, 1])

            im_candidates = self.ent_im_embeddings
            im_candidates = tf.reshape(im_candidates, [1, self.ent_tot, self.dim])
            im_candidates = tf.tile(im_candidates, [b_size, 1, 1])

        else:
            sc_candidates = self.rel_embeddings
            sc_candidates = tf.reshape(sc_candidates, [1, self.rel_tot, self.dim * 2])
            sc_candidates = tf.tile(sc_candidates, [b_size, 1, 1])

            re_candidates = self.rel_re_embeddings
            re_candidates = tf.reshape(re_candidates, [1, self.rel_tot, self.dim])
            re_candidates = tf.tile(re_candidates, [b_size, 1, 1])

            im_candidates = self.rel_im_embeddings
            im_candidates = tf.reshape(im_candidates, [1, self.rel_tot, self.dim])
            im_candidates = tf.tile(im_candidates, [b_size, 1, 1])

        return (sc_h, re_h, im_h), \
               (sc_r, re_r, im_r), \
               (sc_t, re_t, im_t), \
               (sc_candidates, re_candidates, im_candidates)

    def call(self, data):
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
        self.batch_size = int(len(batch_h) / (self.args.neg_triple_num + 1))
        po_score = self.get_pos_score(score)
        ne_score = self.get_neg_score(score)
        loss = self.logistic_loss_tfv2(po_score, ne_score, self.args)
        return loss

    def logistic_loss_tfv2(self, pos_score, neg_score, loss_norm):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.math.reduce_sum(tf.abs(pos_score), axis=1)
            neg_score = tf.math.reduce_sum(tf.abs(neg_score), axis=1)
        else:  # L2 score
            pos_score = tf.math.reduce_sum(tf.square(pos_score), axis=1)
            neg_score = tf.math.reduce_sum(tf.square(neg_score), axis=1)
        pos_loss = tf.math.reduce_sum(tf.math.log(1 + tf.math.exp(pos_score)))
        neg_loss = tf.math.reduce_sum(tf.math.log(1 + tf.math.exp(-neg_score)))
        loss = tf.math.add(pos_loss, neg_loss)
        return loss

    def save(self):
        ent_embeds = self.ent_embeddings.numpy() if self.ent_embeddings is not None else None
        rel_embeds = self.rel_embeddings.numpy() if self.rel_embeddings is not None else None
        read.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=None)
        read.save_special_embeddings(self.out_folder, 'ent_re_embeddings', '', self.ent_re_embeddings.numpy(), None)
        read.save_special_embeddings(self.out_folder, 'ent_im_embeddings', '', self.ent_im_embeddings.numpy(), None)
        read.save_special_embeddings(self.out_folder, 'rel_re_embeddings', '', self.rel_re_embeddings.numpy(), None)
        read.save_special_embeddings(self.out_folder, 'rel_im_embeddings', '', self.rel_im_embeddings.numpy(), None)


"""        
class Analogy(BasicModel):
    '''TransE模型类，定义了TransE的参数空间和loss计算
    '''

    def __init__(self, kgs, args, dim=100):
        super(Analogy, self).__init__(args, kgs)
        self.entity_total = self.ent_tot  # 实体总数
        self.relationship_total = self.rel_tot  # 关系总数
        self.l1_flag = False  # L1正则化
        self.margin = 1.5  # 合页损失函数中的样本差异度值
        self.embedding_dim = dim  # 向量维度
        # 初始化实体语义向量空间
        self.ent_embeddings = tf.keras.layers.Embedding(
            input_dim=self.entity_total, output_dim=self.embedding_dim, name="ent_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )
        # 初始化关系翻译向量空间
        self.rel_embeddings = tf.keras.layers.Embedding(
            input_dim=self.relationship_total, output_dim=self.embedding_dim, name="rel_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )

    def compute_loss(self, data):
        # 计算一个批次数据的合页损失函数值
        # 获得头、尾、关系的 ID
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        pos_h_id, pos_t_id, pos_r_id, neg_h_id, neg_t_id, neg_r_id = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        # 根据ID获得语义向量(E)和转移向量(T)
        pos_h_e = self.ent_embeddings(pos_h_id)
        pos_t_e = self.ent_embeddings(pos_t_id)
        pos_r_e = self.rel_embeddings(pos_r_id)

        neg_h_e = self.ent_embeddings(neg_h_id)
        neg_t_e = self.ent_embeddings(neg_t_id)
        neg_r_e = self.rel_embeddings(neg_r_id)

        if self.l1_flag:
            pos = tf.math.reduce_sum(tf.math.abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.math.reduce_sum(tf.math.abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
        else:
            pos = tf.math.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.math.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
        return tf.math.reduce_sum(tf.math.maximum(pos - neg + self.margin, 0))
"""