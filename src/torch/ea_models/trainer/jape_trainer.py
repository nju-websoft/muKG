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
from src.py.util.util import to_var, to_tensor, task_divide
from src.torch.ea_models.models.jape import JAPE
from src.torch.ea_models.trainer.attr2vec_trainer import attr2vec_trainer
from src.torch.kge_models.basic_model import align_model_trainer


def early_stop(flag1, flag2, flag):
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False


class jape_trainer(align_model_trainer):
    def __init__(self):
        super(jape_trainer, self).__init__()
        self.attr2vec = None
        self.ref_entities2 = None
        self.ref_entities1 = None
        self.attr_sim_mat = None
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = None
        self.optimizer = None

    def init(self, args, kgs):
        self.args = args
        self.kgs = kgs
        if self.args.is_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.model = JAPE(args, kgs)
       
        self.attr2vec = attr2vec_trainer(args, kgs)
        self.model.init()
        self.model.to(self.device)
        self.ref_entities1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_entities2 = self.kgs.valid_entities2 + self.kgs.test_entities2

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
            trained_samples_num += len(batch_pos)
            batch_loss = self.model.define_embed_graph({'pos_hs': to_var([x[0] for x in batch_pos], self.device),
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

    def run_attr2vec(self):
        t = time.time()
        print("Training attribute embeddings:")
        self.attr2vec.run()
        sim_mat = self.attr2vec.get_sim_mat()
        sim_mat[sim_mat < self.args.attr_sim_mat_threshold] = 0
        self.attr_sim_mat = sim_mat
        print("Training attributes ends. Total time = {:.3f} s.".format(time.time() - t))

    def launch_sim_1epo(self, epoch):
        t = time.time()
        steps = len(self.ref_entities1) // self.args.sub_mat_size
        ref_ent1_array = np.array(self.ref_entities1)
        ll = list(range(len(self.ref_entities1)))
        loss = 0
        for i in range(steps):
            idx = random.sample(ll, self.args.sub_mat_size)
            self.optimizer.zero_grad()
            vals = self.model.define_sim_graph({'entities1': to_var(ref_ent1_array[idx], self.device),
                                                'sim_mat': to_tensor(self.attr_sim_mat[idx, :], self.device)})
            vals.backward()
            self.optimizer.step()
            loss += vals
        print('epoch {}, sim loss: {:.4f}, cost time: {:.4f}s'.format(epoch, loss, time.time() - t))

    def test(self):
        rest_12 = self.model.tests(self.kgs.test_entities1, self.kgs.test_entities2)

    def run(self):
        self.run_attr2vec()
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        self.optimizer = get_optimizer_torch(self.args.optimizer, self.model, self.args.learning_rate)
        train_steps = int(math.ceil(triples_num / self.args.batch_size))
        for i in range(1, self.args.max_epoch + 1):
            self.launch_triple_training_1epo(i, triple_steps, steps_tasks, training_batch_queue, None, None)
            self.launch_sim_1epo(i)
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.model.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.args.no_early:
                    self.early_stop = False
                if self.early_stop or i == self.args.max_epoch:
                    break
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
        self.model.save()
