import math
import random
import time

import numpy as np
from joblib._multiprocessing_helpers import mp
from torch.autograd import Variable
import torch
import torch.nn as nn
import pandas as pd

from src.py.base.optimizers import get_optimizer_torch
from src.py.load import batch
from src.py.util.util import to_tensor, task_divide
from src.torch.ea_models.models.imuse import interactive_model, IMUSE
from src.torch.kge_models.basic_model import align_model_trainer


def early_stop(flag1, flag2, flag):
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False


def to_var(batch):
    #batch = batch.astype(int)
    return Variable(torch.from_numpy(batch))


class imuse_trainer(align_model_trainer):
    def __init__(self):
        super(imuse_trainer, self).__init__()
        self.ref_entities2 = None
        self.ref_entities1 = None
        self.aligned_ent_pair_set = None
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = None
        self.optimizer = None

    def init(self, args, kgs):
        self.args = args
        self.kgs = kgs
        self.aligned_ent_pair_set = interactive_model(self.kgs, self.args)
        self.ref_entities1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_entities2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        if self.args.is_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.model = IMUSE(kgs, args)
        self.model.init()
        self.model.to(self.device)

    def launch_training_1epo(self, epoch, relation_triple_steps, relation_step_tasks, relation_batch_queue, triple_steps):
        self.launch_triple_training_1epo(epoch, relation_triple_steps, relation_step_tasks, relation_batch_queue, None, None)
        # self.launch_triple_training_1epo(epoch, triple_steps, data_loader_kg1, data_loader_kg2, neighbors1, neighbors2)
        self.launch_mapping_training_1epo(epoch, triple_steps)

    def launch_triple_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=batch.generate_relation_triple_batch_queue,
                       args=(self.kgs.kg1.relation_triples_list, self.kgs.kg2.relation_triples_list,
                             self.kgs.kg1.relation_triples_set, self.kgs.kg2.relation_triples_set,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2, self.args.neg_triple_num)).start()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            self.optimizer.zero_grad()
            batch_pos, batch_neg = batch_queue.get()
            batch_loss = self.model.define_embed_graph(
                {'pos_hs': to_tensor([x[0] for x in batch_pos], self.device),
                 'pos_rs': to_tensor([x[1] for x in batch_pos], self.device),
                 'pos_ts': to_tensor([x[2] for x in batch_pos], self.device),
                 'neg_hs': to_tensor([x[0] for x in batch_neg], self.device),
                 'neg_rs': to_tensor([x[1] for x in batch_neg], self.device),
                 'neg_ts': to_tensor([x[2] for x in batch_neg], self.device)})
            batch_loss.backward()
            trained_samples_num += len(batch_pos)
            self.optimizer.step()
            epoch_loss += batch_loss.item()
        epoch_loss /= trained_samples_num
        random.shuffle(self.kgs.kg1.relation_triples_list)
        random.shuffle(self.kgs.kg2.relation_triples_list)
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    '''def launch_triple_training_1epo(self, epoch, triple_steps, data_loader_kg1, data_loader_kg2, neighbors1,
                                    neighbors2):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            self.optimizer.zero_grad()
            data1 = data_loader_kg1.__next__()
            data2 = data_loader_kg2.__next__()
            trained_samples_num += (len(data1['pos_hs']) + len(data2['pos_hs']))
            batch_loss = self.model.define_embed_graph(
                {'pos_hs': to_var(np.concatenate((data1['pos_hs'], data2['pos_hs']))),
                 'pos_rs': to_var(np.concatenate((data1['pos_rs'], data2['pos_rs']))),
                 'pos_ts': to_var(np.concatenate((data1['pos_ts'], data2['pos_ts']))),
                 'neg_hs': to_var(np.concatenate((data1['neg_hs'], data2['neg_hs']))),
                 'neg_rs': to_var(np.concatenate((data1['neg_rs'], data2['neg_rs']))),
                 'neg_ts': to_var(np.concatenate((data1['neg_ts'], data2['neg_ts'])))})
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))'''

    def launch_mapping_training_1epo(self, epoch, triple_steps):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(self.aligned_ent_pair_set) / self.args.batch_size))
        for i in range(steps):
            batch_ent_pairs = list(self.aligned_ent_pair_set)
            self.optimizer.zero_grad()
            batch_loss = self.model.align_loss({'aligned_ents1': to_tensor([x[0] for x in batch_ent_pairs], self.device),
                                                'aligned_ents2': to_tensor([x[1] for x in batch_ent_pairs], self.device)})
            batch_loss.backward()
            self.optimizer.step()
            trained_samples_num += len(batch_ent_pairs)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        print('epoch {}, align learning loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def test(self):
        rest_12 = self.model.tests(self.kgs.test_entities1, self.kgs.test_entities2)

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        self.optimizer = get_optimizer_torch(self.args.optimizer, self.model, self.args.learning_rate)
        relation_triples_num = len(self.kgs.kg1.relation_triples_list) + len(self.kgs.kg2.relation_triples_list)
        relation_triple_steps = int(math.ceil(relation_triples_num / self.args.batch_size))
        relation_step_tasks = task_divide(list(range(relation_triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        relation_batch_queue = manager.Queue()
        train_steps = int(math.ceil(triples_num / self.args.batch_size))
        for i in range(1, self.args.max_epoch + 1):
            self.launch_training_1epo(i, relation_triple_steps, relation_step_tasks, relation_batch_queue, train_steps)
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.model.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if i == 800:
                    break
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
        self.test()
        self.model.save()
