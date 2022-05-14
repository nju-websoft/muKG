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
from src.py.util.util import to_var, to_tensor, to_tensor_cpu, task_divide
from src.torch.ea_models.models.attre import AttrE
from src.torch.kge_models.basic_model import align_model_trainer


def early_stop(flag1, flag2, flag):
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False


def formatting_attr_triples(kgs, literal_len):
    """
    Formatting attribute triples from kgs for AttrE.
    :param kgs: modules.load.kgs
    :param literal_len: [optional] Literal truncation length, taking the first literal_len characters.
    :return: attribute_triples_list1_new, attribute_triples_list2_new, char_list size
    """

    def clean_attribute_triples(triples):
        triples_new = []
        for (e, a, v) in triples:
            v = v.split('(')[0].rstrip(' ')
            v = v.replace('.', '').replace('(', '').replace(')', '').replace(',', '') \
                .replace('_', ' ').replace('-', ' ').split('"')[0]
            triples_new.append((e, a, v))
        return triples_new

    attribute_triples_list1 = clean_attribute_triples(kgs.kg1.local_attribute_triples_list)
    attribute_triples_list2 = clean_attribute_triples(kgs.kg2.local_attribute_triples_list)

    value_list = list(set([v for (_, _, v) in attribute_triples_list1 + attribute_triples_list2]))
    char_set = set()
    ch_num = {}
    for literal in value_list:
        for ch in literal:
            n = 1
            if ch in ch_num:
                n += ch_num[ch]
            ch_num[ch] = n

    ch_num = sorted(ch_num.items(), key=lambda x: x[1], reverse=True)
    ch_sum = sum([n for (_, n) in ch_num])
    for i in range(len(ch_num)):
        if ch_num[i][1] / ch_sum >= 0.0001:
            char_set.add(ch_num[i][0])
    char_list = list(char_set)
    char_id_dict = {}
    for i in range(len(char_list)):
        char_id_dict[char_list[i]] = i + 1

    value_char_ids_dict = {}
    for value in value_list:
        char_id_list = [0 for _ in range(literal_len)]
        for i in range(min(len(value), literal_len)):
            if value[i] in char_set:
                char_id_list[i] = char_id_dict[value[i]]
        value_char_ids_dict[value] = char_id_list

    attribute_triples_list1_new, attribute_triples_list2_new = list(), list()
    value_id_char_ids = list()
    value_id_cnt = 0
    for (e_id, a_id, v) in attribute_triples_list1:
        attribute_triples_list1_new.append((e_id, a_id, value_id_cnt))
        value_id_char_ids.append(value_char_ids_dict[v])
        value_id_cnt += 1

    for (e_id, a_id, v) in attribute_triples_list2:
        attribute_triples_list2_new.append((e_id, a_id, value_id_cnt))
        value_id_char_ids.append(value_char_ids_dict[v])
        value_id_cnt += 1
    return attribute_triples_list1_new, attribute_triples_list2_new, to_tensor_cpu(value_id_char_ids), len(char_list) + 1


class attre_trainer(align_model_trainer):
    def __init__(self):
        super(attre_trainer, self).__init__()
        self.char_list_size = None
        self.value_id_char_ids = None
        self.attribute_triples_list2 = None
        self.attribute_triples_list1 = None
        self.model = None
        self.kgs = None
        self.args = None
        self.optimizer2 = None
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
        self.model = AttrE(args, kgs)
        self.attribute_triples_list1, self.attribute_triples_list2, self.value_id_char_ids, self.char_list_size = \
            formatting_attr_triples(self.kgs, self.args.literal_len)
        self.model.initial(self.char_list_size)
        self.model.to(self.device)

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
            trained_samples_num += (len(batch_pos))
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

    def launch_triple_training_1epo_ce(self, epoch, batch_queue, steps_tasks, triple_steps, data_loader_kg1,  data_loader_kg2, neighbors1, neighbors2):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=batch.generate_attribute_triple_batch_queue,
                       args=(self.attribute_triples_list1, self.attribute_triples_list2,
                             set(self.attribute_triples_list1), set(self.attribute_triples_list2),
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, None, None, 1, True)).start()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            self.optimizer.zero_grad()
            batch_pos, batch_neg = batch_queue.get()
            pv1 = torch.index_select(self.value_id_char_ids, 0, to_tensor_cpu([x[2] for x in batch_pos])).to(self.device)
            nv1 = torch.index_select(self.value_id_char_ids, 0, to_tensor_cpu([x[2] for x in batch_neg])).to(self.device)
            batch_loss = self.model.generate_attribute_loss({'pos_hs': to_tensor([x[0] for x in batch_pos], self.device),
                                                             'pos_rs': to_tensor([x[1] for x in batch_pos], self.device),
                                                             'pos_ts': pv1,
                                                             'neg_hs': to_tensor([x[0] for x in batch_neg], self.device),
                                                             'neg_rs': to_tensor([x[1] for x in batch_neg], self.device),
                                                             'neg_ts': nv1})
            trained_samples_num += len(batch_pos)
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
        random.shuffle(self.attribute_triples_list1)
        random.shuffle(self.attribute_triples_list2)
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. attr loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_joint_training_1epo(self, epoch, entities):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(entities) / self.args.batch_size))
        for i in range(steps):
            self.optimizer.zero_grad()
            batch_ents = list(entities)
            # batch_ents = random.sample(batch_ents, self.args.batch_size)
            batch_loss = self.model.generate_align_loss(to_var(batch_ents, self.device))
            batch_loss.backward()
            self.optimizer.step()
            trained_samples_num += len(batch_ents)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        print('epoch {}, joint learning loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def test(self):
        self.model.test(to_tensor(self.kgs.test_entities1, self.device), to_tensor(self.kgs.test_entities2, self.device))

    def run(self):
        t = time.time()
        relation_triples_num = len(self.kgs.kg1.relation_triples_list) + len(self.kgs.kg2.relation_triples_list)
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        attribute_triples_num = len(self.attribute_triples_list1) + len(self.attribute_triples_list2)
        relation_triple_steps = int(math.ceil(relation_triples_num / self.args.batch_size))
        attribute_triple_steps = int(math.ceil(attribute_triples_num / self.args.batch_size))
        entity_list = list(self.kgs.kg1.entities_list + self.kgs.kg2.entities_list)
        attribute_step_tasks = task_divide(list(range(attribute_triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        attribute_batch_queue = manager.Queue()
        relation_step_tasks = task_divide(list(range(relation_triple_steps)), self.args.batch_threads_num)
        relation_batch_queue = manager.Queue()
        self.optimizer = get_optimizer_torch(self.args.optimizer, self.model, self.args.learning_rate)
        # self.optimizer2 = get_optimizer('SGD', self.model, self.args.learning_rate)
        for i in range(1, self.args.max_epoch + 1):
            self.launch_triple_training_1epo(i, relation_triple_steps, relation_step_tasks, relation_batch_queue, None,
                                             None)
            self.launch_triple_training_1epo_ce(i, attribute_batch_queue, attribute_step_tasks, attribute_triple_steps, None, None, None, None)
            self.launch_joint_training_1epo(i, entity_list)
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

