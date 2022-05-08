import math
import time
import os
import numpy as np
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from joblib._multiprocessing_helpers import mp
import tensorflow as tf
from tqdm import tqdm
from src.py.base.losses import get_loss_func_torch, get_loss_func_tf, get_loss_func_tfv2
from src.py.base.optimizers import get_optimizer_torch, get_optimizer_tf
from src.py.evaluation.evaluation import LinkPredictionEvaluator
from src.py.load import batch
from src.py.util.util import task_divide, early_stop, to_var, to_tensor
"""
import ray
# import ray.train as train
from ray import train
import ray.train.torch
from ray.train.trainer import Trainer
"""
from typing import Dict


def get_optimizer(optimizer, learning_rate):
    if optimizer == 'Adam':
        return Adam(learning_rate=learning_rate)
    elif optimizer == 'Adagrad':
        return Adagrad(learning_rate=learning_rate)
    else:
        return SGD(learning_rate=learning_rate)


class kge_trainer:
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
        self.valid = LinkPredictionEvaluator(model, args, kgs, is_valid=True)
        self.optimizer = get_optimizer(self.args.optimizer, self.args.learning_rate)

    def run(self):
        triples_num = self.kgs.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        neighbors1, neighbors2 = None, None
        start = time.time()
        #self.model.compile(optimizer=self.optimizer)
        for i in range(self.args.max_epoch):
            res = 0
            tm = time.time()
            for steps_task in steps_tasks:
                mp.Process(target=batch.generate_relation_triple_batch_queue,
                           args=(self.kgs.relation_triples_list, [],
                                 self.kgs.relation_triples_set, set(),
                                 self.kgs.entities_list, [],
                                 self.args.batch_size, steps_task,
                                 training_batch_queue, neighbors1, neighbors2, self.args.neg_triple_num)).start()
            #print('processing cost time: {:.4f}s'.format(time.time() - tm))

            start = time.time()
            length = 0
            #print(type(self.model))
            for j in range(triple_steps):
                with tf.GradientTape() as tape:
                    batch_pos, batch_neg = training_batch_queue.get()
                    length += len(batch_pos)
                    # batch_pos = np.array(batch_pos)
                    # batch_neg = np.array(batch_neg)
                    datas = np.concatenate((batch_pos, batch_neg), axis=0)

                    data = {
                        'batch_h': np.array([x[0] for x in datas]),
                        'batch_r': np.array([x[1] for x in datas]),
                        'batch_t': np.array([x[2] for x in datas])
                    }
                    loss = self.model(data)
                    """
                    self.batch_size = len(batch_pos)
                    po_score = self.get_pos_score(score)
                    ne_score = self.get_neg_score(score)
                    loss = get_loss_func_tfv2(po_score, ne_score, self.args)
                    """
                #print(self.model.trainable_variables)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                #print(self.model.trainable_variables)
                # 把梯度和变量进行绑定
                grads_and_vars = zip(gradients, self.model.trainable_variables)
                # 进行梯度更新
                self.optimizer.apply_gradients(grads_and_vars)
                #sess = tf.compat.v1.Session()
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

    """
    def run(self):
        triples_num = self.kgs.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        neighbors1, neighbors2 = None, None
        start = time.time()
        print(next(self.model.parameters()).device)
        train_dataset = general_dataloader(self.args, self.kgs)
        data_loader = DataLoader(train_dataset, batch_size=self.args.batch_size)
        for i in range(self.args.max_epoch):
            res = 0
            length = 0
            for data in data_loader:
                self.optimizer.zero_grad()
                self.batch_size = data[0].shape[0]
                length += self.batch_size
                # batch_pos = np.array(batch_pos)
                # batch_neg = np.array(batch_neg)
                # data[3] = to_tensor(data[3], device)
                data = {
                    'batch_h': torch.cat((data[0].squeeze(), data[3].squeeze()), 0).int(),
                    'batch_r': torch.cat((data[1].squeeze(), data[4].squeeze()), 0).int(),
                    'batch_t': torch.cat((data[2].squeeze(), data[5].squeeze()), 0).int()
                }
                score = self.model(data)
                # self.batch_size = len(batch_pos)
                po_score = self.get_pos_score(score)
                ne_score = self.get_neg_score(score)
                loss = get_loss_func_torch(po_score, ne_score, self.args)
                loss.backward()
                self.optimizer.step()
                res += loss.item()
            print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(i, res / length, time.time() - start))
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                t1 = time.time()
                flag = self.valid.print_results()
                print('valid cost time: {:.4f}s'.format(time.time() - start))
                '''self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break'''
        self.save()
        """
    def test(self):
        predict = LinkPredictionEvaluator(self.model, self.args, self.kgs)
        predict.print_results()

    def retest(self):
        self.model.load_embeddings()
        t1 = time.time()
        predict = LinkPredictionEvaluator(self.model, self.args, self.kgs)
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

"""
def get_pos_score(score, batch_size):
    tmp = score[:batch_size]
    return tmp.view(batch_size, -1)


def get_neg_score(score, batch_size):
    tmp = score[batch_size:]
    return tmp.view(batch_size, -1)


def general_dataloader(args, kgs):
    triples_num = kgs.relation_triples_num
    triple_steps = int(math.ceil(triples_num / args.batch_size))
    steps_tasks = task_divide(list(range(triple_steps)), args.batch_threads_num)
    manager = mp.Manager()
    batch_queue = manager.Queue()
    for steps_task in steps_tasks:
        mp.Process(target=batch.generate_relation_triple_batch_queue,
                   args=(kgs.relation_triples_list, [],
                         kgs.relation_triples_set, set(),
                         kgs.entities_list, [],
                         args.batch_size, steps_task,
                         batch_queue, None, None, args.neg_triple_num)).start()
    pos_batch = list()
    neg_batch = list()
    for i in range(triple_steps):
        batch_pos, batch_neg = batch_queue.get()
        pos_batch += batch_pos
        neg_batch += batch_neg
    return ParallelDataset(pos_batch, neg_batch, args.neg_triple_num)


def trainer(config: Dict):
    global early_stop
    args = config["args"]
    kgs = config["kgs"]
    model = config["model"]
    # model.module.generate()
    if args.is_gpu:
        torch.cuda.set_device(3)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cuda:2')
    else:
        device = torch.device('cpu')

    model = train.torch.prepare_model(model)
    optimizer = get_optimizer_torch(args.optimizer, model, args.learning_rate)
    flag1 = -1
    flag2 = -1

    train_dataset = general_dataloader(args, kgs)
    worker_batch_size = args.batch_size // train.world_size()
    data_loader = DataLoader(train_dataset, batch_size=worker_batch_size)
    '''data_loader1 = LoadDataset(kgs.kg1, kg1_size, args.batch_threads_num,
                                args.neg_triple_num)
    data_loader2 = LoadDataset(kgs.kg2, kg2_size, args.batch_threads_num,
                               args.neg_triple_num)'''
    data_loader = train.torch.prepare_data_loader(data_loader)
    t = time.time()
    for i in range(1, args.max_epoch + 1):
        # data_loader_kg1 = iter(data_loader)
        res = 0
        start = time.time()
        length = 0
        for data in data_loader:
            optimizer.zero_grad()
            batch_size = data[0].shape[0]
            length += batch_size
            # batch_pos = np.array(batch_pos)
            # batch_neg = np.array(batch_neg)
            # data[3] = to_tensor(data[3], device)
            data = {
                'batch_h': torch.cat((data[0].squeeze(), data[3].squeeze()), 0),
                'batch_r': torch.cat((data[1].squeeze(), data[4].squeeze()), 0),
                'batch_t': torch.cat((data[2].squeeze(), data[5].squeeze()), 0)
            }
            ts = time.time()
            score = model(data)
            print("length is {}".format(batch_size))
            print("time is {}".format(time.time() - ts))
            po_score = get_pos_score(score, batch_size)
            ne_score = get_neg_score(score, batch_size)
            loss = get_loss_func_torch(po_score, ne_score, args)
            loss.backward()
            optimizer.step()
            res += loss.item()
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(i, res / length, time.time() - start))
        if i >= args.start_valid and i % args.eval_freq == 0:
            t1 = time.time()
            # flag = valid.print_results()
            # print('valid cost time: {:.4f}s'.format(time.time() - start))
    print("Training ends. Total time = {:.3f} s.".format(time.time() - t))

    # print(f"Loss results: {result}")


class parallel_trainer(parallel_model):
    def __init__(self):
        super(parallel_trainer, self).__init__()
        self.kgs = None
        self.args = None
        self.early_stop = None
        self.flag2 = -1
        self.flag1 = -1
        self.NetworkActor = None

    def run(self):
        ray.init(num_cpus=12)
        t = time.time()
        self.train_fashion_mnist()

    def train_fashion_mnist(self):
        trainer1 = Trainer(backend="torch", num_workers=2, use_gpu=False, resources_per_worker={"CPU":2})
        trainer1.start()
        result = trainer1.run(
            train_func=trainer,
            config={"args": self.args, "kgs": self.kgs, "model": self.model},
            # callbacks=[],
        )
        trainer1.shutdown()


class ParallelDataset(Dataset):
    def __init__(self, pos_batch, neg_batch, neg_num):  # add parameters here
        self.pos = pos_batch
        self.neg = neg_batch
        assert neg_num == 1
        self.neg_num = neg_num - 1
        self.batch_size = len(self.pos)
        # self.batch = self.pos + self.neg
        # self.__count_htr()

    def __getitem__(self, index):
        '''return self.pos[index][0], self.pos[index][1], self.pos[index][2], \
               [x[0] for x in self.neg[index: index + self.neg_num]],\
               [x[1] for x in self.neg[index: index + self.neg_num]], \
               [x[2] for x in self.neg[index: index + self.neg_num]]'''
        return self.pos[index][0], self.pos[index][1], self.pos[index][2], \
               self.neg[index][0], self.neg[index][1], self.neg[index][2]

    def __len__(self):
        return len(self.pos)"""
