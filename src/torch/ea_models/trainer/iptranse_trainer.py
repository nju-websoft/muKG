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
from src.py.evaluation.alignment import find_alignment
from src.py.load import batch
from src.py.load.kgs import KGs
from src.py.util.util import task_divide, to_var
from src.torch.ea_models.models.ptranse import IPTransE
from src.torch.kge_models.basic_model import align_model_trainer


def generate_triple_batch(triples, batch_size, ents_list):
    if batch_size > len(triples):
        batch_size = len(triples)
    pos_triples = random.sample(triples, batch_size)
    neg_triples = generate_neg_triples_w(pos_triples, ents_list)
    return pos_triples, neg_triples


def generate_neg_paths(pos_paths, rel_list):
    neg_paths = list()
    for (r_x, r_y, r, _) in pos_paths:
        r2 = random.sample(rel_list, 1)[0]
        neg_paths.append((r_x, r_y, r2))
    return neg_paths


def generate_newly_triples(ent1, ent2, w, rt_dict1, hr_dict1):
    newly_triples = set()
    for r, t in rt_dict1.get(ent1, set()):
        newly_triples.add((ent2, r, t, w))
    for h, r in hr_dict1.get(ent1, set()):
        newly_triples.add((h, r, ent2, w))
    return newly_triples


def generate_triples_of_latent_ents(kgs: KGs, ents1, ents2, tr_ws):
    assert len(ents1) == len(ents2)
    newly_triples = set()
    for i in range(len(ents1)):
        newly_triples |= generate_newly_triples(ents1[i], ents2[i], tr_ws[i], kgs.kg1.rt_dict, kgs.kg1.hr_dict)
        newly_triples |= generate_newly_triples(ents2[i], ents1[i], tr_ws[i], kgs.kg2.rt_dict, kgs.kg2.hr_dict)
    print("newly triples: {}".format(len(newly_triples)))
    return newly_triples


def generate_neg_triples_w(pos_triples, ents_list):
    neg_triples = list()
    for (h, r, t, w) in pos_triples:
        h2, r2, t2 = h, r, t
        choice = random.randint(0, 999)
        if choice < 500:
            h2 = random.sample(ents_list, 1)[0]
        elif choice >= 500:
            t2 = random.sample(ents_list, 1)[0]
        neg_triples.append((h2, r2, t2, w))
    return neg_triples


def generate_batch_queue(kgs: KGs, paths1, paths2, batch_size, path_batch_size, steps, neg_triple_num, out_queue):
    for step in steps:
        pos_triples, neg_triples, pos_paths1, neg_paths1 = generate_batch(kgs, paths1, paths2, batch_size,
                                                                          path_batch_size, step, neg_triple_num)
        out_queue.put((pos_triples, neg_triples, pos_paths1, neg_paths1))


def generate_batch(kgs: KGs, paths1, paths2, batch_size, path_batch_size, step, neg_triple_num):
    pos_triples, neg_triples = batch.generate_relation_triple_batch(kgs.kg1.relation_triples_list,
                                                                  kgs.kg2.relation_triples_list,
                                                                  kgs.kg1.relation_triples_set,
                                                                  kgs.kg2.relation_triples_set,
                                                                  kgs.kg1.entities_list, kgs.kg2.entities_list,
                                                                  batch_size, step,
                                                                  None, None, neg_triple_num)
    num1 = int(len(paths1) / (len(paths1) + len(paths2)) * path_batch_size)
    num2 = path_batch_size - num1
    pos_paths1 = random.sample(paths1, num1)
    pos_paths2 = random.sample(paths2, num2)
    neg_paths1 = generate_neg_paths(pos_paths1, kgs.kg1.relations_list)
    neg_paths2 = generate_neg_paths(pos_paths2, kgs.kg2.relations_list)
    pos_paths1.extend(pos_paths2)
    neg_paths1.extend(neg_paths2)
    return pos_triples, neg_triples, pos_paths1, neg_paths1


def generate_2steps_path(triples):
    tr = np.array([[tr[0], tr[2], tr[1]] for tr in triples])
    tr = pd.DataFrame(tr, columns=['h', 't', 'r'])
    sizes = tr.groupby(['h', 'r']).size()
    sizes.name = 'size'
    tr = tr.join(sizes, on=['h', 'r'])
    train_raw_df = tr[['h', 'r', 't', 'size']]
    two_step_df = pd.merge(train_raw_df, train_raw_df, left_on='t', right_on='h')
    print('start merge triple with path')

    two_step_df['_path_weight'] = two_step_df.size_x * two_step_df.size_y
    two_step_df = two_step_df[two_step_df['_path_weight'] < 101]
    two_step_df = pd.merge(two_step_df, train_raw_df, left_on=['h_x', 't_y'], right_on=['h', 't'], copy=False,
                           sort=False)
    # print(two_step_df[['r_x', 'r_y', 'r', '_path_weight']])
    path_mat = two_step_df[['r_x', 'r_y', 'r', '_path_weight']].values
    print("num of path:", path_mat.shape[0])
    path_list = list()
    for i in range(path_mat.shape[0]):
        path_list.append((int(path_mat[i][0]), int(path_mat[i][1]), int(path_mat[i][2]), float(path_mat[i][3])))
    return path_list


def early_stop(flag1, flag2, flag):
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False


class iptranse_trainer(align_model_trainer):
    def __init__(self):
        super(iptranse_trainer, self).__init__()
        self.device = None
        self.model = None
        self.early_stop = None
        self.optimizer = None
        self.alignment_optimizer = None
        self.ref_entities2 = None
        self.ref_entities1 = None
        self.flag1 = -1
        self.flag2 = -1
        self.paths1, self.paths2 = None, None

    def init(self, args, kgs):
        self.args = args
        self.kgs = kgs
        self.paths1 = generate_2steps_path(self.kgs.kg1.relation_triples_list)
        self.paths2 = generate_2steps_path(self.kgs.kg2.relation_triples_list)
        self.ref_entities1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_entities2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        self.early_stop = None
        if self.args.is_gpu:
            torch.cuda.set_device(2)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.model = IPTransE(args, kgs)
        self.model.init()
        self.model.to(self.device)

    def launch_ptranse_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        path_batch_size = (len(self.paths1) + len(self.paths2)) // triple_steps
        for steps_task in steps_tasks:
            mp.Process(target=generate_batch_queue,
                       args=(self.kgs, self.paths1, self.paths2, self.args.batch_size, path_batch_size,
                             steps_task, self.args.neg_triple_num, batch_queue)).start()
        for i in range(triple_steps):
            self.optimizer.zero_grad()
            pos_triples, neg_triples, pos_paths, neg_paths = batch_queue.get()
            trained_samples_num += (len(pos_triples))
            batch_loss = self.model.generate_transE_loss({'pos_hs': to_var([x[0] for x in pos_triples], self.device),
                                                          'pos_rs': to_var([x[1] for x in pos_triples], self.device),
                                                          'pos_ts': to_var([x[2] for x in pos_triples], self.device),
                                                          'neg_hs': to_var([x[0] for x in neg_triples], self.device),
                                                          'neg_rs': to_var([x[1] for x in neg_triples], self.device),
                                                          'neg_ts': to_var([x[2] for x in neg_triples], self.device),
                                                          'pos_rx': to_var([x[0] for x in pos_paths], self.device),
                                                          'pos_ry': to_var([x[1] for x in pos_paths], self.device),
                                                          'pos_r': to_var([x[2] for x in pos_paths], self.device),
                                                          'neg_rx': to_var([x[0] for x in neg_paths], self.device),
                                                          'neg_ry': to_var([x[1] for x in neg_paths], self.device),
                                                          'neg_r': to_var([x[2] for x in neg_paths], self.device),
                                                          'path_weight': to_var([x[3] for x in pos_paths], self.device)})
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
        epoch_loss /= trained_samples_num
        random.shuffle(self.kgs.kg1.relation_triples_list)
        random.shuffle(self.kgs.kg2.relation_triples_list)
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_alignment_training_1epo(self, epoch):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        sim_mat = self.model.ref_sim_mat()
        pairs = find_alignment(sim_mat, self.args.sim_th, 1)
        if pairs is None or len(pairs) == 0:
            return
        new_ent1 = [self.ref_entities1[pair[0]] for pair in pairs]
        new_ent2 = [self.ref_entities2[pair[1]] for pair in pairs]
        # 将每对备选的实体对的相似程度
        tr_ws = [sim_mat[pair[0], pair[1]] for pair in pairs]
        newly_triples = generate_triples_of_latent_ents(self.kgs, new_ent1, new_ent2, tr_ws)
        steps = math.ceil(((len(newly_triples)) / self.args.batch_size))
        if steps == 0:
            steps = 1
        for i in range(steps):
            newly_pos_batch, newly_neg_batch = generate_triple_batch(newly_triples, self.args.batch_size,
                                                                     self.kgs.kg1.entities_list +
                                                                     self.kgs.kg2.entities_list)
            self.optimizer.zero_grad()
            batch_loss = self.model.generate_align_loss({'pos_hs': to_var(np.array([(x[0]) for x in newly_pos_batch]), self.device),
                                                         'pos_rs': to_var(np.array([(x[1]) for x in newly_pos_batch]), self.device),
                                                         'pos_ts': to_var(np.array([(x[2]) for x in newly_pos_batch]), self.device),
                                                         'neg_hs': to_var(np.array([(x[0]) for x in newly_neg_batch]), self.device),
                                                         'neg_rs': to_var(np.array([(x[1]) for x in newly_neg_batch]), self.device),
                                                         'neg_ts': to_var(np.array([(x[2]) for x in newly_neg_batch]), self.device),
                                                         'path_weight': to_var(np.array([(x[3]) for x in newly_pos_batch]), self.device)})
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
            trained_samples_num += len(newly_pos_batch)
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. mapping loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def test(self):
        rest_12 = self.model.tests(self.kgs.test_entities1, self.kgs.test_entities2)

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        self.optimizer = get_optimizer_torch(self.args.optimizer, self.model, self.args.learning_rate)
        # self.alignment_optimizer = get_optimizer_torch(self.args.optimizer, self.model, self.args.learning_rate)
        for epoch in range(1, self.args.max_epoch):
            self.launch_ptranse_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue)
            if epoch >= self.args.start_valid and epoch % self.args.eval_freq == 0:
                flag = self.model.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or epoch == self.args.max_epoch:
                    break
            if epoch % self.args.bp_freq == 0:
                self.launch_alignment_training_1epo(epoch)
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
        self.test()
        self.model.save()
