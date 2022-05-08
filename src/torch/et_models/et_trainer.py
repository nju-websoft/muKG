import math
import random
import time
import os
from joblib._multiprocessing_helpers import mp
from torch.autograd import Variable
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from src.py.base.losses import get_loss_func_torch
from src.py.base.optimizers import get_optimizer_torch
from src.py.evaluation.evaluation import EntityTypeEvaluator
from src.py.load import batch
from src.py.util.util import task_divide, early_stop, to_var
import ray
# import ray.train as train
from ray import train
import ray.train.torch
from ray.train.trainer import Trainer
from typing import Dict
from src.torch.kge_models.basic_model import parallel_model
from torch.utils.data import DataLoader, Dataset


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"


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
        if self.args.is_gpu:
            #torch.cuda.set_device(1)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # self.device = torch.device('cuda:2')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        self.valid = EntityTypeEvaluator(model, args, kgs, is_valid=True)
        self.optimizer = get_optimizer_torch(self.args.optimizer, self.model, self.args.learning_rate)

    def run(self):
        """
          to_var and to_tensor function needs to be added to_device(gpu)
        """
        triples_num = len(self.kgs.train_et_list)
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        neighbors1, neighbors2 = None, None
        start = time.time()
        print(next(self.model.parameters()).device)
        for i in range(self.args.max_epoch):
            res = 0
            start = time.time()
            length = 0
            for j in range(triple_steps):
                self.optimizer.zero_grad()
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
                    'batch_h': to_var(np.array([x[0] for x in links_batch]), self.device),
                    'batch_r': to_var(np.array(r), self.device),
                    'batch_t': to_var(np.array([x[1] for x in links_batch]), self.device)
                }
                score = self.model(data)
                self.batch_size = int(len(links_batch)/(self.args.neg_triple_num + 1))
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

    def test(self):
        predict = EntityTypeEvaluator(self.model, self.args, self.kgs)
        predict.print_results()

    def retest(self):
        self.model.load_embeddings()
        self.model.to(self.device)
        t1 = time.time()
        predict = EntityTypeEvaluator(self.model, self.args, self.kgs)
        predict.print_results()
        print('test cost time: {:.4f}s'.format(time.time() - t1))

    def get_pos_score(self, score):
        tmp = score[:self.batch_size]
        return tmp.view(self.batch_size, -1)

    def get_neg_score(self, score):
        tmp = score[self.batch_size:]
        return tmp.view(self.batch_size, -1)

    def save(self):
        self.model.save()







