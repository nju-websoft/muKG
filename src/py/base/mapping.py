import numpy as np
import tensorflow._api.v2.compat.v1 as tf

from src.py.base.optimizers import generate_optimizer_tf

tf.disable_eager_execution()  #关闭eager运算

from src.py.base.initializers import orthogonal_init
from src.py.base.losses import mapping_loss_tf


def add_mapping_module(model):
    with tf.name_scope('seed_links_placeholder'):
        model.seed_entities1 = tf.placeholder(tf.int32, shape=[None])
        model.seed_entities2 = tf.placeholder(tf.int32, shape=[None])
    with tf.name_scope('seed_links_lookup'):
        tes1 = tf.nn.embedding_lookup(model.ent_embeds, model.seed_entities1)
        tes2 = tf.nn.embedding_lookup(model.ent_embeds, model.seed_entities2)
    with tf.name_scope('mapping_loss'):
        model.mapping_loss = model.args.alpha * mapping_loss_tf(tes1, tes2, model.mapping_mat, model.eye_mat)
        model.mapping_optimizer = generate_optimizer_tf(model.mapping_loss, model.args.learning_rate,
                                                     opt=model.args.optimizer)


def add_mapping_variables(model):
    with tf.variable_scope('kgs' + 'mapping'):
        model.mapping_mat = orthogonal_init([model.args.dim, model.args.dim], 'mapping_matrix')
        model.eye_mat = tf.constant(np.eye(model.args.dim), dtype=tf.float32, name='eye')
