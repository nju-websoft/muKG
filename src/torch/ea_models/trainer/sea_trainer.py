import math
import random
import time

import numpy as np
from joblib._multiprocessing_helpers import mp
from torch.autograd import Variable
import torch
import torch.nn as nn

from src.py.base.optimizers import get_optimizer_torch
from src.py.load import batch
from src.py.util.util import to_var, task_divide
from src.torch.ea_models.models.sea import SEA
from src.torch.kge_models.basic_model import align_model_trainer


def early_stop(flag1, flag2, flag):
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False


class sea_trainer(align_model_trainer):
    def __init__(self):
        super(sea_trainer, self).__init__()
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = None
        self.optimizer = None

    def init(self, args, kgs):
        self.args = args
        self.kgs = kgs
        self.model = SEA(args, kgs)
        if self.args.is_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.model.init()
        self.model.to(self.device)

    def launch_training_1epo(self, epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2):
        self.launch_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
        self.launch_mapping_training_1epo(epoch, triple_steps)

    def launch_triple_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for steps_task in steps_tasks:
            mp.Process(target=batch.generate_relation_triple_batch_queue,
                       args=(self.kgs.kg1.relation_triples_list, self.kgs.kg2.relation_triples_list,
                             self.kgs.kg1.relation_triples_set, self.kgs.kg2.relation_triples_set,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2, self.args.neg_triple_num)).start()
        for i in range(triple_steps):
            self.optimizer.zero_grad()
            batch_pos, batch_neg = batch_queue.get()
            trained_samples_num += len(batch_neg)
            batch_loss = self.model.generate_transE_loss({'pos_hs': to_var([x[0] for x in batch_pos], self.device),
                                                          'pos_rs': to_var([x[1] for x in batch_pos], self.device),
                                                          'pos_ts': to_var([x[2] for x in batch_pos], self.device),
                                                          'neg_hs': to_var([x[0] for x in batch_neg], self.device),
                                                          'neg_rs': to_var([x[1] for x in batch_neg], self.device),
                                                          'neg_ts': to_var([x[2] for x in batch_neg], self.device)})
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_mapping_training_1epo(self, epoch, triple_steps):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            self.optimizer.zero_grad()
            labeled_batch = random.sample(self.kgs.train_links, len(self.kgs.train_links) // triple_steps)
            unlabeled_batch = random.sample(self.kgs.test_links + self.kgs.valid_links,
                                            len(self.kgs.test_links + self.kgs.valid_links) // triple_steps)
            batch_loss = self.model.generate_mapping_loss({'seed1_labeled': to_var(np.array([x[0] for x in labeled_batch]), self.device),
                                                           'seed2_labeled': to_var(np.array([x[1] for x in labeled_batch]), self.device),
                                                           'seed1_unlabeled': to_var(np.array([x[0] for x in unlabeled_batch]), self.device),
                                                           'seed2_unlabeled': to_var(np.array([x[1] for x in unlabeled_batch]), self.device)})
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
            trained_samples_num += len(labeled_batch)
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. mapping loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def test(self):
        rest_12 = self.model.tests(self.kgs.test_entities1, self.kgs.test_entities2)

    def save(self):
        self.model.save()

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        self.optimizer = get_optimizer_torch(self.args.optimizer, self.model, self.args.learning_rate)
        for i in range(1, self.args.max_epoch + 1):
            self.launch_training_1epo(i, triple_steps, steps_tasks, training_batch_queue, None, None)
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.model.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.args.no_early:
                    self.early_stop = False
                if self.early_stop or i == self.args.max_epoch:
                    break
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
        self.test()
        self.model.save()
