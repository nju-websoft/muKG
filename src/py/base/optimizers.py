

def generate_optimizer_tf(loss, learning_rate, var_list=None, opt='SGD'):
    optimizer = get_optimizer_tf(opt, learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    return optimizer.apply_gradients(grads_and_vars)


def get_optimizer_tf(opt, learning_rate):
    import tensorflow._api.v2.compat.v1 as tf
    # tf.disable_eager_execution()  # 关闭eager运算
    if opt == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif opt == 'Adadelta':
        # To match the exact form in the original paper use 1.0.
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif opt == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    else:  # opt == 'SGD'
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer


def get_optimizer_torch(optimize, model, lr):
    from torch import optim
    if optimize == 'Adagrad':
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=lr
        )
    elif optimize == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr
        )
    return optimizer

