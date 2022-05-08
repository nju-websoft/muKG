def get_loss_func_tf(phs, prs, pts, nhs, nrs, nts, args):
    triple_loss = None
    if args.loss == 'margin-based':
        triple_loss = margin_loss_tf(phs, prs, pts, nhs, nrs, nts, args.margin, args.loss_norm)
    elif args.loss == 'logistic':
        triple_loss = logistic_loss_tf(phs, prs, pts, nhs, nrs, nts, args.loss_norm)
    elif args.loss == 'limited':
        triple_loss = limited_loss_tf(phs, prs, pts, nhs, nrs, nts, args.pos_margin, args.neg_margin, args.loss_norm)
    return triple_loss


def margin_loss_tfv2(pos_score, neg_score, margin, loss_norm):
    import tensorflow as tf
    loss = tf.reduce_sum(tf.nn.relu(margin + pos_score - neg_score))
    return loss


def logistic_loss_tfv2(pos_score, neg_score, loss_norm):
    import tensorflow as tf
    if loss_norm == 'L1':  # L1 score
        pos_score = tf.reduce_sum(tf.abs(pos_score), axis=1)
        neg_score = tf.reduce_sum(tf.abs(neg_score), axis=1)
    else:  # L2 score
        pos_score = tf.reduce_sum(tf.square(pos_score), axis=1)
        neg_score = tf.reduce_sum(tf.square(neg_score), axis=1)
    pos_loss = tf.reduce_sum(tf.math.log(1 + tf.exp(pos_score)))
    neg_loss = tf.reduce_sum(tf.math.log(1 + tf.exp(-neg_score)))
    loss = tf.add(pos_loss, neg_loss)
    return loss


def limited_loss_tfv2(pos_score, neg_score, pos_margin, neg_margin, loss_norm, balance=1.0):
    import tensorflow as tf
    if loss_norm == 'L1':  # L1 score
        pos_score = tf.reduce_sum(tf.abs(pos_score), axis=1)
        neg_score = tf.reduce_sum(tf.abs(neg_score), axis=1)
    else:  # L2 score
        pos_score = tf.reduce_sum(tf.square(pos_score), axis=1)
        neg_score = tf.reduce_sum(tf.square(neg_score), axis=1)
    pos_loss = tf.reduce_sum(tf.nn.relu(pos_score - tf.constant(pos_margin)))
    neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(neg_margin) - neg_score))
    loss = tf.add(pos_loss, balance * neg_loss, name='limited_loss')
    return loss


def get_loss_func_tfv2(pos_score, neg_score, args):
    triple_loss = None
    if args.loss == 'margin-based':
        triple_loss = margin_loss_tfv2(pos_score, neg_score, args.margin, args.loss_norm)
    elif args.loss == 'logistic':
        triple_loss = logistic_loss_tfv2(pos_score, neg_score, args.loss_norm)
    elif args.loss == 'limited':
        triple_loss = limited_loss_tfv2(pos_score, neg_score, args.pos_margin, args.neg_margin, args.loss_norm)
    return triple_loss


def get_loss_func_torch(pos_score, neg_score, args):
    triple_loss = None
    if args.loss == 'margin-based':
        triple_loss = margin_loss_torch(pos_score, neg_score, args.margin)
    elif args.loss == 'logistic':
        triple_loss = logistic_loss_torch(pos_score, neg_score)
    elif args.loss == 'logistic_adv':
        triple_loss = logistic_adv_loss_torch(pos_score, neg_score, args.adv)
    elif args.loss == 'limited':
        triple_loss = limited_loss_torch(pos_score, neg_score,  args.pos_margin, args.neg_margin)
    return triple_loss


def logistic_adv_loss_torch(pos_score, neg_score, adv):
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    adv_temperature = nn.Parameter(torch.Tensor([adv]))
    adv_temperature.requires_grad = False
    weights = F.softmax(neg_score * adv, dim=-1).detach()
    pos_loss = torch.sum(torch.log(1 + torch.exp(pos_score)))
    neg_loss = torch.sum(weights * torch.log(1 + torch.exp(-neg_score)))
    loss = (pos_loss + neg_loss) / 2
    return loss

def margin_loss_tf(phs, prs, pts, nhs, nrs, nts, margin, loss_norm):
    import tensorflow._api.v2.compat.v1 as tf
    tf.disable_eager_execution()  # 关闭eager运算
    tf.disable_v2_behavior()  # 禁用TensorFlow 2.x行为
    with tf.name_scope('margin_loss_distance'):
        pos_distance = phs + prs - pts
        neg_distance = nhs + nrs - nts
        dim = pos_distance.get_shape().as_list()[-1]
    with tf.name_scope('margin_loss'):
        if loss_norm == 'L1':  # L1 normal
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        else:  # L2 normal
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        #batch_size = pos_score.get_shape()
        #print(batch_size)
        #pos_score = tf.reshape(pos_score, [-1, dim])
        #neg_score = tf.reshape(neg_score, [-1, dim])
        loss = tf.reduce_sum(tf.nn.relu(tf.constant(margin) + pos_score - neg_score), name='margin_loss')
    return loss


def margin_loss_torch(pos_score, neg_score, margin):
    import torch
    loss = torch.sum(torch.relu_(margin + pos_score - neg_score))
    return loss


def positive_loss_tf(phs, prs, pts, loss_norm):
    import tensorflow._api.v2.compat.v1 as tf
    tf.disable_eager_execution()  # 关闭eager运算
    with tf.name_scope('positive_loss_distance'):
        pos_distance = phs + prs - pts
    with tf.name_scope('positive_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
        loss = tf.reduce_sum(pos_score, name='positive_loss')
    return loss


def positive_loss_torch(pos_score):
    import torch
    loss = torch.sum(pos_score)
    return loss


def limited_loss_tf(phs, prs, pts, nhs, nrs, nts, pos_margin, neg_margin, loss_norm, balance=1.0):
    import tensorflow._api.v2.compat.v1 as tf
    tf.disable_eager_execution()  # 关闭eager运算
    with tf.name_scope('limited_loss_distance'):
        pos_distance = phs + prs - pts
        neg_distance = nhs + nrs - nts
    with tf.name_scope('limited_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        pos_loss = tf.reduce_sum(tf.nn.relu(pos_score - tf.constant(pos_margin)))
        neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(neg_margin) - neg_score))
        loss = tf.add(pos_loss, balance * neg_loss, name='limited_loss')
    return loss


def limited_loss_torch(pos_score, neg_score, pos_margin, neg_margin, balance=1.0):
    import torch
    pos_loss = torch.sum(torch.relu(pos_score - pos_margin))
    neg_loss = torch.sum(torch.relu(neg_margin - neg_score))
    loss = pos_loss + balance * neg_loss
    return loss


def logistic_loss_tf(phs, prs, pts, nhs, nrs, nts, loss_norm):
    import tensorflow._api.v2.compat.v1 as tf
    tf.disable_eager_execution()  # 关闭eager运算
    with tf.name_scope('logistic_loss_distance'):
        pos_distance = phs + prs - pts
        neg_distance = nhs + nrs - nts
    with tf.name_scope('logistic_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        pos_loss = tf.reduce_sum(tf.log(1 + tf.exp(pos_score)))
        neg_loss = tf.reduce_sum(tf.log(1 + tf.exp(-neg_score)))
        loss = tf.add(pos_loss, neg_loss, name='logistic_loss')
    return loss


def logistic_loss_torch(pos_score, neg_score):
    import torch
    pos_loss = torch.sum(torch.log(1 + torch.exp(pos_score)))
    neg_loss = torch.sum(torch.log(1 + torch.exp(-neg_score)))
    loss = pos_loss + neg_loss
    return loss


def logistic_adv_loss_torch(pos_score, neg_score, adv):
    import torch
    pos_loss = torch.sum(torch.log(1 + torch.exp(pos_score)))
    weights = get_weights(neg_score, adv)
    neg_loss = torch.sum(weights * torch.log(1 + torch.exp(-neg_score)))
    loss = (pos_loss + neg_loss) / 2
    return loss


def get_weights(n_score, adv):
    import torch.nn.functional as F
    return F.softmax(n_score * adv, dim=-1).detach()


def mapping_loss_tf(tes1, tes2, mapping, eye):
    import tensorflow._api.v2.compat.v1 as tf
    tf.disable_eager_execution()  # 关闭eager运算
    mapped_tes2 = tf.matmul(tes1, mapping)
    map_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tes2 - mapped_tes2, 2), 1))
    orthogonal_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tf.matmul(mapping, mapping, transpose_b=True) - eye, 2), 1))
    return map_loss + orthogonal_loss


def mapping_loss_torch(tes1, tes2, mapping, eye):
    import torch
    mapped_tes2 = torch.matmul(tes1, mapping)
    map_loss = torch.sum(torch.norm(tes2 - mapped_tes2, 2, -1), -1)
    orthogonal_loss = torch.sum(torch.sum(torch.pow(torch.matmul(mapping, mapping.t()) - eye, 2), -1))
    return map_loss + orthogonal_loss
