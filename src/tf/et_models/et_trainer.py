import math
import random
import time
import os
from joblib._multiprocessing_helpers import mp
import tensorflow as tf
import numpy as np

from src.py.base.losses import get_loss_func_tfv2
from src.py.evaluation.evaluation import EntityTypeEvaluator
from src.py.load import batch
from src.py.util.util import task_divide
from src.tf.kge_models.kge_trainer import get_optimizer


class et_trainer:
    def __init__(self):
        self.device = None
        self.valid = None
        self.batch_size = None
        self.neg_catch = None
        self.loss = None
        self.data_loader = None
        self.optimizer = None
        self.model = None
        self.kgs = None
        self.args = None
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = None

    def init(self, args, kgs, model):
        self.args = args
        self.kgs = kgs
        self.model = model
        # print(self.model.parameters())
        self.valid = EntityTypeEvaluator(model, args, kgs, is_valid=True)
        self.optimizer = get_optimizer(self.args.optimizer, self.args.learning_rate)

    def run(self):
        triples_num = self.kgs.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        manager = mp.Manager()
        for i in range(self.args.max_epoch):
            res = 0
            start = time.time()
            length = 0
            # print(type(self.model))
            for j in range(triple_steps):
                with tf.GradientTape() as tape:
                    links_batch = random.sample(self.kgs.train_et_list, len(self.kgs.train_et_list) // triple_steps)
                    length += len(links_batch)
                    batch_h = [x[0] for x in links_batch]
                    for k in range(self.args.neg_triple_num):
                        neg_type = np.random.choice(self.kgs.type_list, len(self.kgs.train_et_list) // triple_steps)
                        links_batch_neg = zip(batch_h, neg_type)
                        links_batch += links_batch_neg
                    # batch_pos = np.array(batch_pos)
                    # batch_neg = np.array(batch_neg)
                    r = [0 for x in range(len(links_batch))]
                    data = {
                        'batch_h': np.array([x[0] for x in links_batch]),
                        'batch_r': np.array(r),
                        'batch_t': np.array([x[1] for x in links_batch])
                    }
                    score = self.model(data)

                    self.batch_size = len(batch_h)
                    po_score = self.get_pos_score(score)
                    ne_score = self.get_neg_score(score)
                    loss = get_loss_func_tfv2(po_score, ne_score, self.args)

                # print(self.model.trainable_variables)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                # print(self.model.trainable_variables)
                # 把梯度和变量进行绑定
                grads_and_vars = zip(gradients, self.model.trainable_variables)
                # 进行梯度更新
                self.optimizer.apply_gradients(grads_and_vars)
                # sess = tf.compat.v1.Session()
                res += loss.numpy()
            print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(i, res / length, time.time() - start))
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                t1 = time.time()
                flag = self.valid.print_results()
                print('valid cost time: {:.4f}s'.format(time.time() - start))
                '''self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break'''
        self.save()

    def test(self):
        predict = EntityTypeEvaluator(self.model, self.args, self.kgs)
        predict.print_results()

    def retest(self):
        self.model.load_embeddings()
        t1 = time.time()
        predict = EntityTypeEvaluator(self.model, self.args, self.kgs)
        predict.print_results()
        print('test cost time: {:.4f}s'.format(time.time() - t1))

    def get_pos_score(self, score):
        tmp = score[:self.batch_size]
        return tf.reshape(tmp, [self.batch_size, -1])

    def get_neg_score(self, score):
        tmp = score[self.batch_size:]
        return tf.reshape(tmp, [self.batch_size, -1])

    def save(self):
        self.model.save()







