import gc
import math
import multiprocessing as mp
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.py.evaluation.alignment_finder import find_potential_alignment_mwgm, check_new_alignment
from src.py.evaluation.evaluation import test, valid
from src.py.util.util import to_tensor
from src.torch.kge_models.basic_model import BasicModel


def bootstrapping(sim_mat, unaligned_entities1, unaligned_entities2, labeled_alignment, sim_th, k):
    curr_labeled_alignment = find_potential_alignment_mwgm(sim_mat, sim_th, k)
    if curr_labeled_alignment is not None:
        labeled_alignment = update_labeled_alignment_x(labeled_alignment, curr_labeled_alignment, sim_mat)
        labeled_alignment = update_labeled_alignment_y(labeled_alignment, sim_mat)
        del curr_labeled_alignment
    if labeled_alignment is not None:
        newly_aligned_entities1 = [unaligned_entities1[pair[0]] for pair in labeled_alignment]
        newly_aligned_entities2 = [unaligned_entities2[pair[1]] for pair in labeled_alignment]
    else:
        newly_aligned_entities1, newly_aligned_entities2 = None, None
    del sim_mat
    gc.collect()
    return labeled_alignment, newly_aligned_entities1, newly_aligned_entities2


def update_labeled_alignment_x(pre_labeled_alignment, curr_labeled_alignment, sim_mat):
    labeled_alignment_dict = dict(pre_labeled_alignment)
    n1, n2 = 0, 0
    for i, j in curr_labeled_alignment:
        if labeled_alignment_dict.get(i, -1) == i and j != i:
            n2 += 1
        if i in labeled_alignment_dict.keys():
            pre_j = labeled_alignment_dict.get(i)
            pre_sim = sim_mat[i, pre_j]
            new_sim = sim_mat[i, j]
            if new_sim >= pre_sim:
                if pre_j == i and j != i:
                    n1 += 1
                labeled_alignment_dict[i] = j
        else:
            labeled_alignment_dict[i] = j
    print("update wrongly: ", n1, "greedy update wrongly: ", n2)
    pre_labeled_alignment = set(zip(labeled_alignment_dict.keys(), labeled_alignment_dict.values()))
    check_new_alignment(pre_labeled_alignment, context="after editing (<-)")
    return pre_labeled_alignment


def update_labeled_alignment_y(labeled_alignment, sim_mat):
    labeled_alignment_dict = dict()
    updated_alignment = set()
    for i, j in labeled_alignment:
        i_set = labeled_alignment_dict.get(j, set())
        i_set.add(i)
        labeled_alignment_dict[j] = i_set
    for j, i_set in labeled_alignment_dict.items():
        if len(i_set) == 1:
            for i in i_set:
                updated_alignment.add((i, j))
        else:
            max_i = -1
            max_sim = -10
            for i in i_set:
                if sim_mat[i, j] > max_sim:
                    max_sim = sim_mat[i, j]
                    max_i = i
            updated_alignment.add((max_i, j))
    check_new_alignment(updated_alignment, context="after editing (->)")
    return updated_alignment


def calculate_likelihood_mat(ref_ent1, ref_ent2, labeled_alignment):
    def set2dic(alignment):
        if alignment is None:
            return None
        dic = dict()
        for i, j in alignment:
            dic[i] = j
        assert len(dic) == len(alignment)
        return dic

    t = time.time()
    ref_mat = np.zeros((len(ref_ent1), len(ref_ent2)), dtype=np.float32)
    if labeled_alignment is not None:
        alignment_dic = set2dic(labeled_alignment)
        n = 1 / len(ref_ent1)
        for ii in range(len(ref_ent1)):
            if ii in alignment_dic.keys():
                ref_mat[ii, alignment_dic.get(ii)] = 1
            else:
                for jj in range(len(ref_ent1)):
                    ref_mat[ii, jj] = n
    print("calculate likelihood matrix costs {:.2f} s".format(time.time() - t))
    return ref_mat


def generate_supervised_triples(rt_dict1, hr_dict1, rt_dict2, hr_dict2, ents1, ents2):
    assert len(ents1) == len(ents2)
    newly_triples1, newly_triples2 = list(), list()
    for i in range(len(ents1)):
        newly_triples1.extend(generate_newly_triples(ents1[i], ents2[i], rt_dict1, hr_dict1))
        newly_triples2.extend(generate_newly_triples(ents2[i], ents1[i], rt_dict2, hr_dict2))
    print("newly triples: {}, {}".format(len(newly_triples1), len(newly_triples2)))
    return newly_triples1, newly_triples2


def generate_newly_triples(ent1, ent2, rt_dict1, hr_dict1):
    newly_triples = list()
    for r, t in rt_dict1.get(ent1, set()):
        newly_triples.append((ent2, r, t))
    for h, r in hr_dict1.get(ent1, set()):
        newly_triples.append((h, r, ent2))
    return newly_triples


def generate_pos_batch(triples1, triples2, step, batch_size):
    num1 = int(len(triples1) / (len(triples1) + len(triples2)) * batch_size)
    num2 = batch_size - num1
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > len(triples1):
        end1 = len(triples1)
    if end2 > len(triples2):
        end2 = len(triples2)
    pos_triples1 = triples1[start1: end1]
    pos_triples2 = triples2[start2: end2]
    return pos_triples1, pos_triples2


def mul(tensor1, tensor2, session, num, sigmoid):
    t = time.time()
    if num < 20000:
        sim_mat = torch.matmul(tensor1, tensor2.T)
        if sigmoid:
            res = torch.sigmoid(sim_mat).detach()
        else:
            res = sim_mat.detach()
    else:
        res = np.matmul(tensor1.detach(), tensor2.detach().T)
    print("mat mul costs: {:.3f}".format(time.time() - t))
    return res


class BootEA(BasicModel):

    def __init__(self, kgs, args):
        super().__init__(args, kgs)
        self.margin = None
        self.rel_embeds = None
        self.ent_embeds = None
        self.ref_ent1 = None
        self.kgs = kgs
        self.args = args
        self.ref_ent2 = None
        self.epsilon = None

    def init(self):
        self.ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        self.ent_embeds = nn.Embedding(self.kgs.entities_num, self.args.dim)
        self.rel_embeds = nn.Embedding(self.kgs.relations_num, self.args.dim)
        self.margin = nn.Parameter(torch.Tensor([1.5]))
        self.margin.requires_grad = False
        '''self.epsilon = 2.0
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.dim]),
            requires_grad=False
        )'''
        '''nn.init.uniform_(tensor=self.ent_embeds.weight.data, a=-self.embedding_range.item(),
                         b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_embeds.weight.data, a=-self.embedding_range.item(),
                         b=self.embedding_range.item())'''

        '''nn.init.normal_(self.ent_embeds.weight.data, std=0.1)
        nn.init.normal_(self.rel_embeds.weight.data, std=0.1)'''
        nn.init.xavier_normal_(self.ent_embeds.weight.data)
        nn.init.xavier_normal_(self.rel_embeds.weight.data)

        # self.ent_embeds.weight.data = F.normalize(self.ent_embeds.weight.data, 2, -1)
        # self.rel_embeds.weight.data = F.normalize(self.rel_embeds.weight.data, 2, -1)
        # customize parameters
        assert self.args.init == 'normal'
        assert self.args.alignment_module == 'swapping'
        assert self.args.loss == 'limited'
        assert self.args.neg_sampling == 'truncated'
        # assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        # assert self.args.loss_norm == 'L2'
        self.args.loss_norm = 'L2'
        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True

        assert self.args.pos_margin >= 0.0
        assert self.args.neg_margin > self.args.pos_margin

        assert self.args.neg_triple_num > 1
        assert self.args.truncated_epsilon > 0.0
        assert self.args.learning_rate >= 0.01

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
        # pos = pos.view(batch_size_now, -1)
        # neg = neg.view(batch_size_now, -1)
        pos_loss = torch.sum(torch.relu_(pos - self.args.pos_margin))
        neg_loss = torch.sum(torch.relu_(self.args.neg_margin - neg))
        loss = pos_loss + self.args.neg_margin_balance * neg_loss
        return loss

    def define_alignment_graph(self, data):
        new_h = data['new_h']
        new_r = data['new_r']
        new_t = data['new_t']
        phs = F.normalize(self.ent_embeds(new_h), 2, -1)
        prs = F.normalize(self.rel_embeds(new_r), 2, -1)
        pts = F.normalize(self.ent_embeds(new_t), 2, -1)
        """
        phs = self.ent_embeds(new_h)
        prs = self.rel_embeds(new_r)
        pts = self.ent_embeds(new_t)
        """
        return - torch.sum(torch.log(torch.sigmoid(-torch.pow(torch.norm(phs + prs - pts, 2, -1), 2))))

    def define_likelihood_graph(self, data):
        entities1 = data['entities1']
        entities2 = data['entities2']
        likelihood_mat = data['likelihood_mat']
        ent1_embed = F.normalize(self.ent_embeds(entities1), 2, -1)
        ent2_embed = F.normalize(self.ent_embeds(entities2), 2, -1)
        """
        ent1_embed = self.ent_embeds(entities1)
        ent2_embed = self.ent_embeds(entities2)
        """
        mat = torch.log(torch.sigmoid(torch.matmul(ent1_embed, ent2_embed.T)))
        likelihood_loss = -torch.sum(torch.multiply(mat, likelihood_mat))
        return likelihood_loss

    def eval_ref_sim_mat(self):
        refs1_embeddings = F.normalize(self.ent_embeds(to_tensor(self.ref_ent1, self.device)), 2, -1)
        refs2_embeddings = F.normalize(self.ent_embeds(to_tensor(self.ref_ent2, self.device)), 2, -1)
        return torch.matmul(refs1_embeddings, refs2_embeddings.T).cpu().detach()

    def eval_kg1_useful_ent_embeddings(self):
        embeds = self.ent_embeds(to_tensor(self.kgs.useful_entities_list1, self.device)).cpu().detach().numpy()
        # embeds = self.ent_embeds(to_tensor(self.kgs.useful_entities_list1)).detach()
        return embeds

    def eval_kg2_useful_ent_embeddings(self):
        embeds = self.ent_embeds(to_tensor(self.kgs.useful_entities_list2, self.device)).cpu().detach().numpy()
        # embeds = self.ent_embeds(to_tensor(self.kgs.useful_entities_list2)).detach()
        return embeds

    def test(self, entities1, entities2):
        seed_entity1 = F.normalize(self.ent_embeds(to_tensor(entities1, self.device)), 2, -1)
        seed_entity2 = F.normalize(self.ent_embeds(to_tensor(entities2, self.device)), 2, -1)
        """
        seed_entity1 = self.ent_embeds(to_tensor(entities1, self.device))
        seed_entity2 = self.ent_embeds(to_tensor(entities2, self.device))
        """
        _, _, _, sim_list = test(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(), None,
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm,
                                 csls_k=0, accurate=True)
        print()
        return sim_list

    def valid(self, stop_metric):
        if len(self.kgs.valid_links) > 0:
            seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities1, self.device)), 2, -1)
            seed_entity2 = F.normalize(
                self.ent_embeds(to_tensor(self.kgs.valid_entities2 + self.kgs.test_entities2, self.device)), 2, -1)
            """
            seed_entity1 = self.ent_embeds(to_tensor(self.kgs.valid_entities1, self.device))
            seed_entity2 = self.ent_embeds(to_tensor(self.kgs.valid_entities2 + self.kgs.test_entities2, self.device))
            """
        else:
            '''seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities1)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities2)), 2, -1)'''
            seed_entity1 = self.ent_embeds(to_tensor(self.kgs.test_entities1, self.device))
            seed_entity2 = self.ent_embeds(to_tensor(self.kgs.test_entities2, self.device))
        hits1_12, mrr_12 = valid(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(), None,
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12


"""
def bootstrapping(sim_mat, unaligned_entities1, unaligned_entities2, labeled_alignment, sim_th, k):
    curr_labeled_alignment = find_potential_alignment_mwgm(sim_mat, sim_th, k)
    if curr_labeled_alignment is not None:
        labeled_alignment = update_labeled_alignment_x(labeled_alignment, curr_labeled_alignment, sim_mat)
        labeled_alignment = update_labeled_alignment_y(labeled_alignment, sim_mat)
        del curr_labeled_alignment
    if labeled_alignment is not None:
        newly_aligned_entities1 = [unaligned_entities1[pair[0]] for pair in labeled_alignment]
        newly_aligned_entities2 = [unaligned_entities2[pair[1]] for pair in labeled_alignment]
    else:
        newly_aligned_entities1, newly_aligned_entities2 = None, None
    del sim_mat
    gc.collect()
    return labeled_alignment, newly_aligned_entities1, newly_aligned_entities2


def update_labeled_alignment_x(pre_labeled_alignment, curr_labeled_alignment, sim_mat):
    labeled_alignment_dict = dict(pre_labeled_alignment)
    n1, n2 = 0, 0
    for i, j in curr_labeled_alignment:
        if labeled_alignment_dict.get(i, -1) == i and j != i:
            n2 += 1
        if i in labeled_alignment_dict.keys():
            pre_j = labeled_alignment_dict.get(i)
            pre_sim = sim_mat[i, pre_j]
            new_sim = sim_mat[i, j]
            if new_sim >= pre_sim:
                if pre_j == i and j != i:
                    n1 += 1
                labeled_alignment_dict[i] = j
        else:
            labeled_alignment_dict[i] = j
    print("update wrongly: ", n1, "greedy update wrongly: ", n2)
    pre_labeled_alignment = set(zip(labeled_alignment_dict.keys(), labeled_alignment_dict.values()))
    check_new_alignment(pre_labeled_alignment, context="after editing (<-)")
    return pre_labeled_alignment


def update_labeled_alignment_y(labeled_alignment, sim_mat):
    labeled_alignment_dict = dict()
    updated_alignment = set()
    for i, j in labeled_alignment:
        i_set = labeled_alignment_dict.get(j, set())
        i_set.add(i)
        labeled_alignment_dict[j] = i_set
    for j, i_set in labeled_alignment_dict.items():
        if len(i_set) == 1:
            for i in i_set:
                updated_alignment.add((i, j))
        else:
            max_i = -1
            max_sim = -10
            for i in i_set:
                if sim_mat[i, j] > max_sim:
                    max_sim = sim_mat[i, j]
                    max_i = i
            updated_alignment.add((max_i, j))
    check_new_alignment(updated_alignment, context="after editing (->)")
    return updated_alignment


def calculate_likelihood_mat(ref_ent1, ref_ent2, labeled_alignment):
    def set2dic(alignment):
        if alignment is None:
            return None
        dic = dict()
        for i, j in alignment:
            dic[i] = j
        assert len(dic) == len(alignment)
        return dic

    t = time.time()
    ref_mat = np.zeros((len(ref_ent1), len(ref_ent2)), dtype=np.float32)
    if labeled_alignment is not None:
        alignment_dic = set2dic(labeled_alignment)
        n = 1 / len(ref_ent1)
        for ii in range(len(ref_ent1)):
            if ii in alignment_dic.keys():
                ref_mat[ii, alignment_dic.get(ii)] = 1
            else:
                for jj in range(len(ref_ent1)):
                    ref_mat[ii, jj] = n
    print("calculate likelihood matrix costs {:.2f} s".format(time.time() - t))
    return ref_mat


def generate_supervised_triples(rt_dict1, hr_dict1, rt_dict2, hr_dict2, ents1, ents2):
    assert len(ents1) == len(ents2)
    newly_triples1, newly_triples2 = list(), list()
    for i in range(len(ents1)):
        newly_triples1.extend(generate_newly_triples(ents1[i], ents2[i], rt_dict1, hr_dict1))
        newly_triples2.extend(generate_newly_triples(ents2[i], ents1[i], rt_dict2, hr_dict2))
    print("newly triples: {}, {}".format(len(newly_triples1), len(newly_triples2)))
    return newly_triples1, newly_triples2


def generate_newly_triples(ent1, ent2, rt_dict1, hr_dict1):
    newly_triples = list()
    for r, t in rt_dict1.get(ent1, set()):
        newly_triples.append((ent2, r, t))
    for h, r in hr_dict1.get(ent1, set()):
        newly_triples.append((h, r, ent2))
    return newly_triples


def generate_pos_batch(triples1, triples2, step, batch_size):
    num1 = int(len(triples1) / (len(triples1) + len(triples2)) * batch_size)
    num2 = batch_size - num1
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > len(triples1):
        end1 = len(triples1)
    if end2 > len(triples2):
        end2 = len(triples2)
    pos_triples1 = triples1[start1: end1]
    pos_triples2 = triples2[start2: end2]
    return pos_triples1, pos_triples2


def mul(tensor1, tensor2, session, num, sigmoid):
    t = time.time()
    if num < 20000:
        sim_mat = torch.matmul(tensor1, tensor2.T)
        if sigmoid:
            res = torch.sigmoid(sim_mat).detach()
        else:
            res = sim_mat.detach()
    else:
        res = np.matmul(tensor1.detach(), tensor2.detach().T)
    print("mat mul costs: {:.3f}".format(time.time() - t))
    return res


class BootEA(BasicModel):

    def __init__(self, kgs, args):
        super().__init__(args, kgs)
        self.margin = None
        self.rel_embeds = None
        self.ent_embeds = None
        self.ref_ent1 = None
        self.kgs = kgs
        self.args = args
        self.ref_ent2 = None

    def init(self):
        self.ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        self.ent_embeds = nn.Embedding(self.kgs.entities_num, self.args.dim)
        self.rel_embeds = nn.Embedding(self.kgs.relations_num, self.args.dim)
        self.margin = nn.Parameter(torch.Tensor([1.5]))
        self.margin.requires_grad = False
        '''nn.init.normal_(self.ent_embeds.weight.data, std=0.1)
        nn.init.normal_(self.rel_embeds.weight.data, std=0.1)'''
        nn.init.xavier_normal_(self.ent_embeds.weight.data)
        nn.init.xavier_normal_(self.rel_embeds.weight.data)
        self.ent_embeds.weight.data = F.normalize(self.ent_embeds.weight.data, 2, -1)
        self.rel_embeds.weight.data = F.normalize(self.rel_embeds.weight.data, 2, -1)
        # customize parameters
        assert self.args.init == 'normal'
        assert self.args.alignment_module == 'swapping'
        assert self.args.loss == 'limited'
        assert self.args.neg_sampling == 'truncated'
        # assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        # assert self.args.loss_norm == 'L2'

        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True

        assert self.args.pos_margin >= 0.0
        assert self.args.neg_margin > self.args.pos_margin

        assert self.args.neg_triple_num > 1
        assert self.args.truncated_epsilon > 0.0
        assert self.args.learning_rate >= 0.01

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
        '''ph = self.ent_embeds(ph)
        pr = self.rel_embeds(pr)
        pt = self.ent_embeds(pt)
        nh = self.ent_embeds(nh)
        nr = self.rel_embeds(nr)
        nt = self.ent_embeds(nt)'''
        if self.args.loss_norm == "L2":
            pos = torch.norm(ph + pr - pt, 2, -1)
            neg = torch.norm(nh + nr - nt, 2, -1)
        else:
            pos = torch.norm(ph + pr - pt, 1, -1)
            neg = torch.norm(nh + nr - nt, 1, -1)
        # pos = pos.view(batch_size_now, -1)
        # neg = neg.view(batch_size_now, -1)
        pos_loss = torch.sum(torch.relu_(pos - self.args.pos_margin))
        neg_loss = torch.sum(torch.relu_(self.args.neg_margin - neg))
        loss = pos_loss + self.args.neg_margin_balance * neg_loss
        return loss

    def define_alignment_graph(self, data):
        new_h = data['new_h']
        new_r = data['new_r']
        new_t = data['new_t']
        phs = F.normalize(self.ent_embeds(new_h), 2, -1)
        prs = F.normalize(self.rel_embeds(new_r), 2, -1)
        pts = F.normalize(self.ent_embeds(new_t), 2, -1)
        '''phs = self.ent_embeds(new_h)
        prs = self.rel_embeds(new_r)
        pts = self.ent_embeds(new_t)'''
        return - torch.sum(torch.log(torch.sigmoid(-torch.norm(phs + prs - pts, 2, -1))))

    def define_likelihood_graph(self, data):
        entities1 = data['entities1']
        entities2 = data['entities2']
        likelihood_mat = data['likelihood_mat']
        ent1_embed = F.normalize(self.ent_embeds(entities1), 2, -1)
        ent2_embed = F.normalize(self.ent_embeds(entities2), 2, -1)
        '''ent1_embed = self.ent_embeds(entities1)
        ent2_embed = self.ent_embeds(entities2)'''
        mat = torch.log(torch.sigmoid(torch.matmul(ent1_embed, ent2_embed.T)))
        likelihood_loss = -torch.sum(torch.multiply(mat, likelihood_mat))
        return likelihood_loss

    def eval_ref_sim_mat(self):
        refs1_embeddings = F.normalize(self.ent_embeds(to_tensor(self.ref_ent1, self.device)), 2, -1).cpu().detach()
        refs2_embeddings = F.normalize(self.ent_embeds(to_tensor(self.ref_ent2, self.device)), 2, -1).cpu().detach()
        return np.matmul(refs1_embeddings, refs2_embeddings.T)

    def eval_kg1_useful_ent_embeddings(self):
        embeds = F.normalize(self.ent_embeds(to_tensor(self.kgs.useful_entities_list1, self.device)), 2, -1).cpu().detach()
        # embeds = self.ent_embeds(to_tensor(self.kgs.useful_entities_list1)).detach()
        return embeds

    def eval_kg2_useful_ent_embeddings(self):
        embeds = F.normalize(self.ent_embeds(to_tensor(self.kgs.useful_entities_list2, self.device)), 2, -1).cpu().detach()
        # embeds = self.ent_embeds(to_tensor(self.kgs.useful_entities_list2)).detach()
        return embeds

    def test(self, entities1, entities2):
        '''seed_entity1 = F.normalize(self.ent_embeds(to_tensor(entities1)), 2, -1)
        seed_entity2 = F.normalize(self.ent_embeds(to_tensor(entities2)), 2, -1)'''
        seed_entity1 = self.ent_embeds(to_tensor(entities1, self.device))
        seed_entity2 = self.ent_embeds(to_tensor(entities2, self.device))
        _, _, _, sim_list = test(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(), None,
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm,
                                 csls_k=0, accurate=True)
        print()
        return sim_list

    def valid(self, stop_metric):
        if len(self.kgs.valid_links) > 0:
            '''seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities1)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities2)), 2, -1)'''
            seed_entity1 = self.ent_embeds(to_tensor(self.kgs.valid_entities1, self.device))
            seed_entity2 = self.ent_embeds(to_tensor(self.kgs.valid_entities2, self.device))
        else:
            '''seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities1)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities2)), 2, -1)'''
            seed_entity1 = self.ent_embeds(to_tensor(self.kgs.test_entities1, self.device))
            seed_entity2 = self.ent_embeds(to_tensor(self.kgs.test_entities2, self.device))
        hits1_12, mrr_12 = valid(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(), None,
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12
"""