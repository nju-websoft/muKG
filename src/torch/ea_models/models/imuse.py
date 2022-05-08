import torch
import torch.nn as nn
import Levenshtein
import time
import math
import multiprocessing as mp
import multiprocessing
import torch.nn.functional as F

from src.py.evaluation.evaluation import test, valid
from src.py.util.util import to_tensor
from src.torch.kge_models.basic_model import BasicModel


def interactive_model(kgs, args):
    start = time.time()
    aligned_ent_pair_set_all = set()
    i = 0
    aligned_attr_pair_set_all = get_aligned_attr_pair_by_name_similarity(kgs, 0.6)
    print('aligned_attr_pair_set:', len(aligned_attr_pair_set_all))
    while True:
        i += 1
        aligned_ent_pair_set_iter = align_entity_by_attributes(kgs, aligned_attr_pair_set_all, args.sim_thresholds_ent)
        aligned_ent_pair_set_all |= aligned_ent_pair_set_iter
        print(i, 'len(aligned_ent_pair_set_all):', len(aligned_ent_pair_set_all), 'len(aligned_ent_pair_set_iter):',
              len(aligned_ent_pair_set_iter))
        if i >= args.interactive_model_iter_num:
            break
        aligned_attr_pair_set_iter = align_attribute_by_entities(kgs, aligned_ent_pair_set_all,
                                                                 args.sim_thresholds_attr)
        if len(aligned_attr_pair_set_all | aligned_attr_pair_set_iter) == len(aligned_attr_pair_set_all):
            break
        aligned_attr_pair_set_all |= aligned_attr_pair_set_iter
        print(i, 'len(aligned_attr_pair_set_all):', len(aligned_attr_pair_set_all), 'len(aligned_attr_pair_set_iter):',
              len(aligned_attr_pair_set_iter))
    print(time.time() - start)
    return aligned_ent_pair_set_all


def run_one_ea(ent_attrs_dict_1, ent_attrs_dict_2, ent_attr_value_dict_1, ent_attr_value_dict_2, sim_thresholds_ent,
               aligned_attr_pair_set):
    aligned_ent_pair_set_i = set()
    cnt = 0
    target_ent_set = set()
    for e1, attrs1 in ent_attrs_dict_1.items():
        cnt += 1
        target_ent = None
        sim_max = sim_thresholds_ent
        for e2, attrs2 in ent_attrs_dict_2.items():
            sim, sim_cnt = 0, 0
            for (a1, a2) in aligned_attr_pair_set:
                if a1 in attrs1 and a2 in attrs2:
                    sim += compute_two_values_similarity(ent_attr_value_dict_1[(e1, a1)],
                                                         ent_attr_value_dict_2[(e2, a2)])
                    sim_cnt += 1
            if sim_cnt > 0:
                sim /= sim_cnt
            if sim > sim_max:
                target_ent = e2
                sim_max = sim
            if target_ent is not None and target_ent not in target_ent_set:
                aligned_ent_pair_set_i.add((e1, target_ent))
                target_ent_set.add(target_ent)
    return aligned_ent_pair_set_i


def align_entity_by_attributes(kgs, aligned_attr_pair_set, sim_thresholds_ent):
    print('align_entity_by_attributes...')
    aligned_ent_pair_set = set()
    if len(aligned_attr_pair_set) == 0:
        return aligned_ent_pair_set
    ent_attrs_dict_1, ent_attr_value_dict_1 = filter_by_aligned_attributes(kgs.kg1.attribute_triples_set,
                                                                           set([a for (a, _) in aligned_attr_pair_set]))
    ent_attrs_dict_2, ent_attr_value_dict_2 = filter_by_aligned_attributes(kgs.kg2.attribute_triples_set,
                                                                           set([a for (_, a) in aligned_attr_pair_set]))
    ent_set_1 = list(ent_attrs_dict_1.keys())
    size = len(ent_set_1) // 8
    pool = multiprocessing.Pool(processes=8)
    res = list()
    for i in range(8):
        if i == 7:
            ent_set_i = ent_set_1[size * i:]
        else:
            ent_set_i = ent_set_1[size * i:size * (i + 1)]
        ent_attrs_dict_1_i = dict([(k, ent_attrs_dict_1[k]) for k in ent_set_i])
        res.append(pool.apply_async(run_one_ea, (ent_attrs_dict_1_i, ent_attrs_dict_2, ent_attr_value_dict_1,
                                                 ent_attr_value_dict_2, sim_thresholds_ent, aligned_attr_pair_set)))
    pool.close()
    pool.join()

    for _res in res:
        aligned_ent_pair_set |= _res.get()
    temp_dict = dict([(x, y) for (x, y) in aligned_ent_pair_set])
    aligned_ent_pair_set = set([(x, y) for x, y in temp_dict.items()])
    return aligned_ent_pair_set


def run_one_ae(attr_ents_dict_1, attr_ents_dict_2, attr_ent_value_dict_1, attr_ent_value_dict_2, sim_thresholds_attr,
               aligned_ent_pair_set):
    aligned_attr_pair_set = set()
    target_attr_set = set()
    for a1, ents1 in attr_ents_dict_1.items():
        target_attr = None
        sim_max = sim_thresholds_attr
        for a2, ents2 in attr_ents_dict_2.items():
            sim, sim_cnt = 0, 0
            for (e1, e2) in aligned_ent_pair_set:
                if e1 in ents1 and e2 in ents2:
                    sim += compute_two_values_similarity(attr_ent_value_dict_1[(a1, e1)],
                                                         attr_ent_value_dict_2[(a2, e2)])
                    sim_cnt += 1
            if sim_cnt > 0:
                sim /= sim_cnt
            if sim > sim_max:
                target_attr = a2
                sim_max = sim
            if target_attr is not None and target_attr not in target_attr_set:
                aligned_attr_pair_set.add((a1, target_attr))
                target_attr_set.add(target_attr)
    return aligned_attr_pair_set


def align_attribute_by_entities(kgs, aligned_ent_pair_set, sim_thresholds_attr):
    print('align_attribute_by_entities...')
    aligned_attr_pair_set = set()
    if aligned_ent_pair_set is None or len(aligned_ent_pair_set) == 0:
        return aligned_attr_pair_set
    attr_ents_dict_1, attr_ent_value_dict_1 = filter_by_aligned_attributes(kgs.kg1.attribute_triples_set,
                                                                           set([e for (e, _) in aligned_ent_pair_set]))
    attr_ents_dict_2, attr_ent_value_dict_2 = filter_by_aligned_attributes(kgs.kg2.attribute_triples_set,
                                                                           set([e for (_, e) in aligned_ent_pair_set]))
    attr_set_1 = list(attr_ents_dict_1.keys())
    size = len(attr_set_1) // 8
    pool = multiprocessing.Pool(processes=8)
    res = list()
    for i in range(8):
        if i == 7:
            attr_set_i = attr_set_1[size * i:]
        else:
            attr_set_i = attr_set_1[size * i:size * (i + 1)]
        attr_ents_dict_1_i = dict([(k, attr_ents_dict_1[k]) for k in attr_set_i])
        res.append(pool.apply_async(run_one_ae, (attr_ents_dict_1_i, attr_ents_dict_2, attr_ent_value_dict_1,
                                                 attr_ent_value_dict_2, sim_thresholds_attr, aligned_ent_pair_set)))
    pool.close()
    pool.join()

    for _res in res:
        aligned_attr_pair_set |= _res.get()
    temp_dict = dict([(x, y) for (x, y) in aligned_attr_pair_set])
    aligned_attr_pair_set = set([(x, y) for x, y in temp_dict.items()])
    return aligned_attr_pair_set


def filter_by_aligned_attributes(attr_triples, attr_set):
    ent_attrs_dict, ent_attr_value_dict = {}, {}
    for (e, a, v) in attr_triples:
        if a in attr_set and (e, a) not in ent_attr_value_dict:
            ent_attr_value_dict[(e, a)] = v
            attrs = set()
            if e in ent_attrs_dict:
                attrs = ent_attrs_dict[e]
            attrs.add(a)
            ent_attrs_dict[e] = attrs
    return ent_attrs_dict, ent_attr_value_dict


def filter_by_aligned_entities(attr_triples, ent_set):
    attr_ents_dict, attr_ent_value_dict = {}, {}
    for (e, a, v) in attr_triples:
        if e in ent_set and (a, e) not in attr_ent_value_dict:
            attr_ent_value_dict[(a, e)] = v
            ents = set()
            if a in attr_ents_dict:
                ents = attr_ents_dict[a]
            attr_ents_dict[e] = ents
    return attr_ents_dict, attr_ent_value_dict


def cal_lcs_sim(first_str, second_str):
    len_1 = len(first_str.strip())
    len_2 = len(second_str.strip())
    len_vv = [[0] * (len_2 + 2)] * (len_1 + 2)
    for i in range(1, len_1 + 1):
        for j in range(1, len_2 + 1):
            if first_str[i - 1] == second_str[j - 1]:
                len_vv[i][j] = 1 + len_vv[i - 1][j - 1]
            else:
                len_vv[i][j] = max(len_vv[i - 1][j], len_vv[i][j - 1])

    return float(float(len_vv[len_1][len_2] * 2) / float(len_1 + len_2))


def compute_two_values_similarity(v1, v2):
    # lcs_sim = cal_lcs_sim(v1, v2)
    # return lcs_sim/(Levenshtein.ratio(v1, v2)+lcs_sim)*2
    return Levenshtein.ratio(v1, v2)


def get_aligned_attr_pair_by_name_similarity(kgs, sim_thresholds_attr, top_k=10):
    def turn_id_attr_dict(attr_id_dict):
        id_attr_dict = {}
        for a, i in attr_id_dict.items():
            id_attr_dict[i] = a
        return id_attr_dict

    id_attr_dict_1 = turn_id_attr_dict(kgs.kg1.attributes_id_dict)
    id_attr_dict_2 = turn_id_attr_dict(kgs.kg2.attributes_id_dict)
    aligned_attr_pair_set = set()
    attr2_set = set()
    for attr1 in kgs.kg1.attributes_set:
        target_attr = None
        sim_max = sim_thresholds_attr
        attr_str_1 = id_attr_dict_1[attr1].split('/')[-1]
        for attr2 in kgs.kg2.attributes_set:
            attr_str_2 = id_attr_dict_2[attr2].split('/')[-1]
            sim = Levenshtein.ratio(attr_str_1, attr_str_2)
            if sim > sim_max:
                target_attr = attr2
                sim_max = sim
        if target_attr is not None and target_attr not in attr2_set:
            aligned_attr_pair_set.add((attr1, target_attr))
            attr2_set.add(target_attr)

    attr_num_dict_1, attr_num_dict_2 = {}, {}
    for (_, a, _) in kgs.kg1.attribute_triples_set:
        num = 1
        if a in attr_num_dict_1:
            num += attr_num_dict_1[a]
        attr_num_dict_1[a] = num
    for (_, a, _) in kgs.kg2.attribute_triples_set:
        num = 1
        if a in attr_num_dict_2:
            num += attr_num_dict_2[a]
        attr_num_dict_2[a] = num
    attr_pair_num_dict = {}
    for (a1, a2) in aligned_attr_pair_set:
        num = 0
        if a1 in attr_num_dict_1:
            num += attr_num_dict_1[a1]
        if a2 in attr_num_dict_2:
            num += attr_num_dict_2[a2]
        attr_pair_num_dict[(a1, a2)] = num
    attr_pair_list = sorted(attr_pair_num_dict.items(), key=lambda d: d[1], reverse=True)
    if top_k > len(attr_pair_list):
        top_k = len(attr_pair_list)
    aligned_attr_pair_set_top = set([a_pair for (a_pair, _) in attr_pair_list[: top_k]])
    return aligned_attr_pair_set_top


class IMUSE(BasicModel):

    def __init__(self, kgs, args):
        super().__init__(args, kgs)

    def init(self):
        self._define_variables()

        # customize parameters
        assert self.args.init == 'normal'
        assert self.args.loss == 'margin_based'
        assert self.args.neg_sampling == 'uniform'
        # assert self.args.optimizer == 'SGD'
        assert self.args.eval_metric == 'inner'
        self.args.loss_norm = 'L2'

        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True
        assert self.args.learning_rate >= 0.01

    def _define_variables(self):
        self.ent_embeds = nn.Embedding(self.kgs.entities_num, self.args.dim)
        self.rel_embeds = nn.Embedding(self.kgs.relations_num, self.args.dim)
        self.margin = nn.Parameter(torch.Tensor([1.5]))
        self.margin.requires_grad = False
        if self.args.init == 'xavier':
            nn.init.xavier_uniform_(self.ent_embeds.weight.data)
            nn.init.xavier_uniform_(self.rel_embeds.weight.data)
        elif self.args.init == 'normal':
            std = 1.0 / math.sqrt(self.args.dim)
            nn.init.trunc_normal_(self.ent_embeds.weight.data, std=std)
            nn.init.trunc_normal_(self.rel_embeds.weight.data, std=std)
        elif self.args.init == 'uniform':
            nn.init.uniform_(self.ent_embeds.weight.data, 0)
            nn.init.uniform_(self.rel_embeds.weight.data, 0)
        #self.ent_embeds.weight.data = F.normalize(self.ent_embeds.weight.data, 2, -1)
        #self.rel_embeds.weight.data = F.normalize(self.rel_embeds.weight.data, 2, -1)

    def define_embed_graph(self, data):
        ph = data['pos_hs']
        pr = data['pos_rs']
        pt = data['pos_ts']
        nh = data['neg_hs']
        nr = data['neg_rs']
        nt = data['neg_ts']
        batch_size_now = ph.shape[0]
        ph = F.normalize(self.ent_embeds(ph), 2, -1)
        pr = F.normalize(self.rel_embeds(pr), 2, -1)
        pt = F.normalize(self.ent_embeds(pt), 2, -1)
        nh = F.normalize(self.ent_embeds(nh), 2, -1)
        nr = F.normalize(self.rel_embeds(nr), 2, -1)
        nt = F.normalize(self.ent_embeds(nt), 2, -1)
        """
        ph = self.ent_embeds(ph)
        pr = self.rel_embeds(pr)
        pt = self.ent_embeds(pt)
        nh = self.ent_embeds(nh)
        nr = self.rel_embeds(nr)
        nt = self.ent_embeds(nt)
        """
        if self.args.loss_norm == "L2":
            pos = torch.pow(torch.norm(ph + pr - pt, 2, -1), 2)
            neg = torch.pow(torch.norm(nh + nr - nt, 2, -1), 2)
        else:
            pos = torch.norm(ph + pr - pt, 1, -1)
            neg = torch.norm(nh + nr - nt, 1, -1)
        pos = pos.view(batch_size_now, -1)
        neg = neg.view(batch_size_now, -1)
        return torch.sum(torch.relu_(pos - neg + self.margin))

    def align_loss(self, data):
        '''ents1 = F.normalize(self.ent_embeds(data['aligned_ents1']), 2, -1)
        ents2 = F.normalize(self.ent_embeds(data['aligned_ents2']), 2, -1)'''
        ents1 = self.ent_embeds(data['aligned_ents1'])
        ents2 = self.ent_embeds(data['aligned_ents2'])
        align_loss = torch.sum(torch.pow(torch.norm(ents1 - ents2, 2, -1), 2))
        return align_loss

    def tests(self, entities1, entities2):
        seed_entity1 = F.normalize(self.ent_embeds(to_tensor(entities1, self.device)), 2, -1)
        seed_entity2 = F.normalize(self.ent_embeds(to_tensor(entities2, self.device)), 2, -1)
        '''seed_entity1 = self.ent_embeds(to_tensor(entities1))
        seed_entity2 = self.ent_embeds(to_tensor(entities2))'''
        sim_list = test(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(), None,
                         self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                         normalize=self.args.eval_norm,
                         csls_k=0, accurate=True)
        print()
        return sim_list

    def valid(self, stop_metric):
        if len(self.kgs.valid_links) > 0:
            '''seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities1, self.device)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities2, self.device)), 2, -1)'''
            seed_entity1 = self.ent_embeds(to_tensor(self.kgs.valid_entities1, self.device))
            seed_entity2 = self.ent_embeds(to_tensor(self.kgs.valid_entities2 + self.kgs.test_entities2, self.device))
        else:
            seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities1, self.device)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities2, self.device)), 2, -1)
            '''seed_entity1 = self.ent_embeds(to_tensor(self.kgs.test_entities1))
            seed_entity2 = self.ent_embeds(to_tensor(self.kgs.test_entities2))'''
        hits1_12, mrr_12 = valid(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(), None,
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12
