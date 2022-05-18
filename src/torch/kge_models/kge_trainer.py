import math
import time
import os
import tqdm
from src.torch.kge_models.pytorch_dataloader import PyTorchTrainDataset
from joblib._multiprocessing_helpers import mp
from torch.autograd import Variable
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.py.base.losses import get_loss_func_torch
from src.py.base.optimizers import get_optimizer_torch
from src.py.evaluation.evaluation import LinkPredictionEvaluator
from src.py.load import batch
from src.py.util.util import task_divide, early_stop, to_var, to_tensor
import ray
# import ray.train as train
from ray import train
import ray.train.torch
from ray.train.trainer import Trainer
from typing import Dict

from src.torch.kge_models.basic_model import parallel_model


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
        if self.args.is_gpu:
            # torch.cuda.set_device(0)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # self.device = torch.device('cuda:2')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        self.valid = LinkPredictionEvaluator(model, args, kgs, is_valid=True)
        self.optimizer = get_optimizer_torch(self.args.optimizer, self.model, self.args.learning_rate)
        train_dataset = PyTorchTrainDataset(self.kgs.relation_triples_list, self.args.neg_triple_num, kgs)
        self.data_loader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                                      collate_fn=train_dataset.collate_fn,
                                      shuffle=True, pin_memory=True, num_workers=self.args.batch_threads_num,
                                      drop_last=False)

    def run_t(self):
        triples_num = self.kgs.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        neighbors1, neighbors2 = None, None
        start = time.time()
        print(next(self.model.parameters()).device)
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
            # print('processing cost time: {:.4f}s'.format(time.time() - tm))

            start = time.time()
            length = 0
            for j in range(triple_steps):
                self.optimizer.zero_grad()
                batch_pos, batch_neg = training_batch_queue.get()
                self.batch_size = len(batch_pos)
                # print(len(batch_neg))
                # length += len(batch_pos)
                batch_pos = np.array(batch_pos)
                batch_neg = np.array(batch_neg)
                datas = np.concatenate((batch_pos, batch_neg), axis=0)
                # datas = batch_pos
                data = {
                    'batch_h': to_var(np.array([x[0] for x in datas]), self.device),
                    'batch_r': to_var(np.array([x[1] for x in datas]), self.device),
                    'batch_t': to_var(np.array([x[2] for x in datas]), self.device),
                }
                score = self.model(data)

                length += self.batch_size
                po_score = self.get_pos_score(score)
                ne_score = self.get_neg_score(score)
                loss = get_loss_func_torch(po_score, ne_score, self.args)
                loss.backward()
                self.optimizer.step()
                res += loss.item()
                """
                score.backward()
                self.optimizer.step()
                length = length + 1
                res += score.item()
                """
            print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(i, res / length, time.time() - start))
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                t1 = time.time()
                flag = self.valid.print_results()
                print('valid cost time: {:.4f}s'.format(time.time() - start))
                '''self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break'''
        self.save()

    def run(self):
        print(next(self.model.parameters()).device)
        for i in range(self.args.max_epoch):
            res = 0
            length = 0
            start = time.time()
            for data in self.data_loader:
                self.optimizer.zero_grad()
                self.batch_size = int(data[0].shape[0] / (self.args.neg_triple_num + 1))
                # print(len(data[0]))
                # batch_pos = np.array(batch_pos)
                # batch_neg = np.array(batch_neg)
                # data[3] = to_tensor(data[3], device)
                data = {
                    'batch_h': data[0].to(self.device),
                    'batch_r': data[1].to(self.device),
                    'batch_t': data[2].to(self.device)
                }
                score = self.model(data)
                if self.model.__class__.__name__ == 'ConvE' or self.model.__class__.__name__ == 'TuckER':
                    score.backward()
                    self.optimizer.step()
                    length = length + 1
                    res += score.item()
                    continue
                length += self.batch_size
                po_score = self.get_pos_score(score)
                ne_score = self.get_neg_score(score)
                # print(po_score)
                loss = get_loss_func_torch(po_score, ne_score, self.args)
                loss.backward()
                self.optimizer.step()
                res += loss.item()
                # time.sleep(0.003)
                # self.batch_size = len(batch_pos)
            print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(i, res / length, time.time() - start))
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                t1 = time.time()
                flag = self.valid.print_results()
                print('valid cost time: {:.4f}s'.format(time.time() - start))
                # TODO: Add early stop for KGE here.
                '''self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break'''
        self.test()
        self.save()

    def test(self):
        predict = LinkPredictionEvaluator(self.model, self.args, self.kgs)
        predict.print_results()

    def retest(self):
        if self.model.__class__.__name__ == 'ConvE':
            self.model = torch.load(self.model.out_folder + 'conve.pth')
        else:
            self.model.load_embeddings()
        self.model.to(self.device)
        t1 = time.time()
        predict = LinkPredictionEvaluator(self.model, self.args, self.kgs)
        predict.print_results()
        print('test cost time: {:.4f}s'.format(time.time() - t1))

    def get_pos_score(self, score):
        tmp = score[:self.batch_size]
        return tmp.view(self.batch_size, -1)

    def get_neg_score(self, score):
        tmp = score[self.batch_size:]
        # print(tmp.view(self.batch_size, -1).shape)
        return tmp.view(self.batch_size, -1)

    def save(self):
        if self.model.__class__.__name__ == 'ConvE':
            if not os.path.exists(self.model.out_folder):
                os.makedirs(self.model.out_folder)
            # print(self.state_dict())
            torch.save(self.model, self.model.out_folder + 'conve.pth')
            #self.model = torch.load(self.model.out_folder + 'conve.pth')
        else:
            self.model.save()


def get_pos_score(score, batch_size):
    tmp = score[:batch_size]
    return tmp.view(batch_size, -1)


def get_neg_score(score, batch_size):
    tmp = score[batch_size:]
    return tmp.view(batch_size, -1)


def trainer(config: Dict):
    global early_stop
    args = config["args"]
    kgs = config["kgs"]
    model = config["model"]
    # model = nn.Linear(4, 1)
    # model.module.generate()
    if args.is_gpu:
        # torch.cuda.set_device(3)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cuda:2')
    else:
        device = torch.device('cpu')

    model = train.torch.prepare_model(model)
    valid = LinkPredictionEvaluator(model.module, args, kgs, is_valid=True)
    # model = train.torch.prepare_model(model)
    optimizer = get_optimizer_torch(args.optimizer, model, args.learning_rate)
    flag1 = -1
    flag2 = -1

    train_dataset = PyTorchTrainDataset(kgs.relation_triples_list, args.neg_triple_num, kgs)
    worker_batch_size = args.batch_size * args.num_worker // train.world_size()
    data_loader = DataLoader(train_dataset, batch_size=worker_batch_size,
                             collate_fn=train_dataset.collate_fn, shuffle=True,
                             pin_memory=True, num_workers=10)
    data_loader = train.torch.prepare_data_loader(data_loader)
    t = time.time()
    for i in range(1, args.max_epoch + 1):
        # data_loader_kg1 = iter(data_loader)
        res = 0
        start = time.time()
        length = 0
        for data in data_loader:
            optimizer.zero_grad()
            data0 = data[0]
            data1 = data[1]
            data2 = data[2]
            # print(len(data[0]))
            data = {
                'batch_h': data0,
                'batch_r': data1,
                'batch_t': data2
            }
            score = model(data)
            if model.mudule.__class__.__name__ == 'ConvE' or model.module.__class__.__name__ == 'TuckER':
                length += 1
                score.backward()
                optimizer.step()
                res += score.item()
                continue
            batch_size = int(data0.shape[0] / (args.neg_triple_num + 1))
            po_score = get_pos_score(score, batch_size)
            ne_score = get_neg_score(score, batch_size)
            loss = get_loss_func_torch(po_score, ne_score, args)
            loss.backward()
            optimizer.step()
            res += loss.item()
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(i, res / length, time.time() - start))
        if i >= args.start_valid and i % args.eval_freq == 0:
            t1 = time.time()
            flag = valid.print_results()
            # print('valid cost time: {:.4f}s'.format(time.time() - start))
    print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
    predict = LinkPredictionEvaluator(model.module, args, kgs)
    predict.print_results()
    model.module.save()

    # print(f"Loss results: {result}")


class parallel_trainer(parallel_model):
    """Provides multi-process and multi-GPU parallel training for KGE models, inheriting class parallel_model

        Parameters
        ----------
        args: dict
            A python dict from muKG.src.py.args. It stored detailed information about model
            training and testing.
        kg: muKG.src.py.KG
            Store the whole information of a KG, like h_dict, r_dict, t_dict,
            train_dataset, valid_dataset, test_dataset and so on.
    """
    def __init__(self):
        super(parallel_trainer, self).__init__()
        self.kgs = None
        self.args = None
        self.early_stop = None
        self.flag2 = -1
        self.flag1 = -1
        self.NetworkActor = None

    def run(self):
        """Initialize ray with number of GPU or CPU.
        """
        self.args.device_number = min(self.args.num_worker, self.args.device_number)
        if self.args.is_gpu:
            ray.init(num_gpus=self.args.device_number)
        else:
            ray.init(num_cpus=self.args.device_number)
        self.train_fashion_mnist()

    def train_fashion_mnist(self):
        """
        Activate ray train by allocating device number and worker number.
        """
        device_allocate = self.args.device_number / self.args.num_worker
        device_allocate = min(device_allocate, 1)
        if self.args.is_gpu:
            trainer1 = Trainer(backend="torch", num_workers=self.args.num_worker, use_gpu=self.args.is_gpu,
                               resources_per_worker={"GPU": device_allocate})
        else:
            trainer1 = Trainer(backend="torch", num_workers=self.args.num_worker, use_gpu=self.args.is_gpu,
                               resources_per_worker={"CPU": device_allocate})
        trainer1.start()
        trainer1.run(
            train_func=trainer,
            config={"args": self.args, "kgs": self.kgs, "model": self.model},
        )
        trainer1.shutdown()

    def test(self):
        pass
