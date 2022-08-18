import math
import random

import numpy as np
#tf.disable_eager_execution()  #关闭eager运算
#tf.disable_v2_behavior()    #禁用TensorFlow 2.x行为
import tensorflow as tf


def init_embeddings(shape, name, init, is_l2_norm, dtype=tf.float32):
    """Initialize embeddings for ent and rel.

        Parameters
        ----------
        shape: list
            The int list specifies dims of embedding for initializing.
        init: string
            This parameter specifies four ways to initialize embeddings, eg:xavier, normal, uniform and unit.
        is_l2_norm: bool
            If is_l2_norm is true, it will return an embedding with l2 normalize after initializing.
        Returns
        -------
        embedding: torch.nn.Embedding
    """
    embeds = None
    if init == 'xavier':
        embeds = xavier_init(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'normal':
        embeds = truncated_normal_init(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'uniform':
        embeds = random_uniform_init(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'unit':
        embeds = random_unit_init(shape, name, is_l2_norm, dtype=dtype)
    return embeds


def xavier_init(shape, name, is_l2_norm, dtype=None):
    import tensorflow._api.v2.compat.v1 as tf
    with tf.name_scope('xavier_init'):
        embeddings = tf.get_variable(name, shape=shape, dtype=dtype,
                                     initializer=tf.keras.initializers.glorot_normal())
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def truncated_normal_init(shape, name, is_l2_norm, dtype=None):
    import tensorflow._api.v2.compat.v1 as tf
    with tf.name_scope('truncated_normal'):
        std = 1.0 / math.sqrt(shape[1])
        embeddings = tf.get_variable(name, shape=shape, dtype=dtype,
                                     initializer=tf.initializers.truncated_normal(stddev=std))
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def random_uniform_init(shape, name, is_l2_norm, minval=0, maxval=None, dtype=None):
    import tensorflow._api.v2.compat.v1 as tf
    #tf.disable_eager_execution()  # 关闭eager运算
    # tf.disable_v2_behavior()    #禁用TensorFlow 2.x行为
    with tf.name_scope('random_uniform'):
        embeddings = tf.get_variable(name, shape=shape, dtype=dtype,
                                     initializer=tf.initializers.random_uniform(minval=minval, maxval=maxval))
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def random_unit_init(shape, name, is_l2_norm, dtype=None):
    import tensorflow._api.v2.compat.v1 as tf
    from sklearn import preprocessing
    #tf.disable_eager_execution()  # 关闭eager运算
    # tf.disable_v2_behavior()    #禁用TensorFlow 2.x行为
    with tf.name_scope('random_unit_init'):
        vectors = list()
        for i in range(shape[0]):
            vectors.append([random.gauss(0, 1) for j in range(shape[1])])
    embeddings = tf.Variable(preprocessing.normalize(np.matrix(vectors)), name=name, dtype=dtype)
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def orthogonal_init(shape, name, dtype=None):
    import tensorflow._api.v2.compat.v1 as tf
    #tf.disable_eager_execution()  # 关闭eager运算
    # tf.disable_v2_behavior()    #禁用TensorFlow 2.x行为
    with tf.name_scope('orthogonal_init'):
        embeddings = tf.get_variable(name, shape=shape, dtype=dtype, initializer=tf.initializers.orthogonal())
    return embeddings
