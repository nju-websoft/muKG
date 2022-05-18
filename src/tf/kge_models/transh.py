import tensorflow as tf
from tqdm import tqdm

from src.py.base.initializers import init_embeddings
from src.py.base.losses import get_loss_func_tf, get_loss_func_tfv2
from src.py.load import read
import numpy as np
tf.compat.v1.enable_eager_execution()
from src.tf.kge_models.basic_model import BasicModel


class TransH(BasicModel):

    def __init__(self, kgs, args):
        super(TransH, self).__init__(args, kgs)

        self.evaluated_projections = False
        self.dim = self.args.dim
        self.margin = self.args.margin
        # self.epsilon = epsilon
        self.p_norm = 1

        self.ent_embeddings = init_embeddings([self.ent_tot, self.dim], 'ent_embeddings',
                                              'xavier', False, dtype=tf.double)
        # 初始化关系翻译向量空间
        self.rel_embeddings = init_embeddings([self.rel_tot, self.dim], 'rel_embeddings',
                                              'xavier', False, dtype=tf.double)

        self.norm_vector = init_embeddings([self.rel_tot, self.dim], 'norm_vector',
                                              'xavier', False, dtype=tf.double)
        self.projected = False
        self.projected_entities = np.zeros(shape=(self.rel_tot, self.ent_tot, self.dim))

    def calc(self, h, r, t):
        if self.p_norm == 'L1':
            score = tf.math.reduce_sum(tf.math.abs(h + r - t), -1)
        else:
            score = tf.math.reduce_sum((h + r - t) ** 2, -1)
        # neg = tf.math.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
        # return tf.math.reduce_sum(tf.math.maximum(pos - neg + self.margin, 0))
        return score

    def evaluate_projections(self):
        """Link prediction evaluation helper function. Project all entities
        according to each relation. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        """
        if self.projected:
            return

        for i in tqdm(range(self.ent_tot), unit='entities', desc='Projecting entities'):

            norm_vect = self.norm_vector

            ent = tf.nn.embedding_lookup(self.ent_embeddings, i)
            norm_components = tf.reduce_sum(tf.reshape(ent, [1, -1]) * norm_vect, -1)

            self.projected_entities[:, i, :] = tf.reshape(ent, [1, -1]) - tf.reshape(norm_components, [-1, 1]) * norm_vect

            del norm_components

        self.projected = True

    def transfer(self, e, norm):
        norm = tf.nn.l2_normalize(norm, -1)
        return e - tf.reshape(tf.reduce_sum(e * norm, 1), [-1, 1]) * norm

    def get_score(self, h, r, t):
        return self.calc(h, r, t)

    def get_embeddings(self, hid, rid, tid, mode = 'entity'):
        self.evaluate_projections()
        r_embs = tf.expand_dims(tf.nn.embedding_lookup(self.rel_embeddings, rid), 1)

        if mode == 'entity':
            proj_h = tf.expand_dims(self.projected_entities[rid, hid], 1)  # shape: (b_size, 1, emb_dim)
            proj_t = tf.expand_dims(self.projected_entities[rid, tid], 1)  # shape: (b_size, 1, emb_dim)
            candidates = self.projected_entities[rid]  # shape: (b_size, self.n_rel, self.emb_dim)
            return proj_h, r_embs, proj_t, candidates
        else:
            proj_h = self.projected_entities[:, hid].transpose(0, 1)  # shape: (b_size, n_rel, emb_dim)
            b_size = proj_h.shape[0]
            proj_t = self.projected_entities[:, tid].transpose(0, 1)  # shape: (b_size, n_rel, emb_dim)
            candidates = self.rel_embeddings.weight.data.view(1, self.rel_tot, self.emb_dim)
            candidates = candidates.expand(b_size, self.rel_tot, self.emb_dim)
            return proj_h, r_embs, proj_t, candidates

    def call(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = tf.nn.embedding_lookup(self.ent_embeddings, batch_h)
        t = tf.nn.embedding_lookup(self.ent_embeddings, batch_t)
        r = tf.nn.embedding_lookup(self.rel_embeddings, batch_r)
        r_norm = tf.nn.embedding_lookup(self.norm_vector, batch_r)
        h = self.transfer(h, r_norm)
        t = self.transfer(t, r_norm)
        score = tf.reshape(self.calc(h, r, t), [-1])
        #self.batch_size = int(len(batch_h) / (self.args.neg_triple_num + 1))
        #po_score = self.get_pos_score(score)
        #ne_score = self.get_neg_score(score)
        #score = get_loss_func_tfv2(po_score, ne_score, self.args)
        # score = tf.math.reduce_sum(tf.math.maximum(po_score - ne_score + self.margin, 0))
        return score

    def save(self):
        ent_embeds = self.ent_embeddings.numpy()
        rel_embeds = self.rel_embeddings.numpy()
        read.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=None)
        norm_vector = self.norm_vector.numpy()
        read.save_special_embeddings(self.out_folder, 'norm_vector', '', norm_vector, None)
