import math
import multiprocessing as mp
import random
import sys
import time
import gc
import numpy as np
import os

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.trainer.util import generate_out_folder
from src.py.base.losses import get_loss_func_torch
from src.py.evaluation.evaluation import valid, test, entity_alignment_evaluation
from src.py.load import read
from src.py.util.env_checker import module_exists
from src.py.util.util import task_divide, to_tensor


def generate_out_folder(out_folder, training_data_path, div_path, method_name):
    params = training_data_path.strip('/').split('/')
    print(out_folder, training_data_path, params, div_path, method_name)
    path = params[-1]
    if module_exists():
        envs = "torch"
    else:
        envs = "tf"
    folder = out_folder + method_name + '/' + path + "/" + div_path + "/" + envs + "/"
    print("results output folder:", folder)
    return folder


class BasicModel(nn.Module):
    def init(self):
        # need to be overwrite
        pass

    def __init__(self, args, kgs):
        super(BasicModel, self).__init__()
        self.type_embeddings = None
        self.transfer_matrix = None
        self.rel_im_embeddings = None
        self.rel_re_embeddings = None
        self.ent_im_embeddings = None
        self.ent_re_embeddings = None
        self.norm_vector = None
        self.rel_transfer = None
        self.ent_transfer = None
        self.ent_embeddings = None
        self.rel_embeddings = None
        self.batch_size = None
        self.low_values = True
        self.args = args
        self.kgs = kgs
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division,
                                              self.__class__.__name__)
        if self.args.is_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.ent_tot = self.kgs.entities_num
        self.rel_tot = self.kgs.relations_num
        self.seed_entities1 = None
        self.seed_entities2 = None
        self.neg_ts = None
        self.neg_rs = None
        self.neg_hs = None
        self.pos_ts = None
        self.pos_rs = None
        self.pos_hs = None
        self.mapping_mat = None
        self.eye_mat = None

        self.triple_optimizer = None
        self.triple_loss = None
        self.mapping_optimizer = None
        self.mapping_loss = None

        self.mapping_matrix = None
        self.ent_npy = None
        self.rel_npy = None
        self.attr_npy = None
        self.map_npy = None
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = False

    def _define_variables(self):
        self.ent_embeddings = nn.Embedding(self.kgs.entities_num, self.args.dim)
        self.rel_embeddings = nn.Embedding(self.kgs.relations_num, self.args.dim)
        nn.init.xavier_uniform_(self.ent_embeds.weight.data)
        nn.init.xavier_uniform_(self.rel_embeds.weight.data)

    def eval_valid(self):
        ent_embeds = self.ent_embeddings.weight.data
        rel_embeds = self.rel_embeddings.weight.data
        return ent_embeds, rel_embeds

    def save(self):
        mapping_mat = self.mapping_matrix.cpu().detach().numpy() if self.mapping_matrix is not None else None
        if self.ent_embeddings is not None:
            ent_embeds = self.ent_embeddings.cpu().weight.data
            rel_embeds = self.rel_embeddings.cpu().weight.data
            read.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat)
        else:
            ent_embeds = self.ent_embeds.cpu().weight.data
            rel_embeds = self.rel_embeds.cpu().weight.data
            read.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat)

    def load_embeddings(self):
        """
        This function we used for link prediction, firstly we load embeddings,
        the the evaluation class can simply pass the h, r, t ids to this function,
        the model returns the score.
        """
        dir = self.out_folder.split("/")
        new_dir = ""
        print(dir)
        for i in range(len(dir) - 1):
            new_dir += (dir[i] + "/")
        exist_file = os.listdir(new_dir)
        new_dir = new_dir + "/"
        if self.__class__.__name__ == 'ComplEx':
            ent_re_embeddings = np.load(new_dir + "ent_re_embeddings.npy")
            ent_im_embeddings = np.load(new_dir + "ent_im_embeddings.npy")
            rel_re_embeddings = np.load(new_dir + "rel_re_embeddings.npy")
            rel_im_embeddings = np.load(new_dir + "rel_im_embeddings.npy")
            self.ent_re_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(ent_re_embeddings))
            self.ent_im_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(ent_im_embeddings))
            self.rel_re_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(rel_re_embeddings))
            self.rel_im_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(rel_im_embeddings))
            return
        ent_embeds = np.load(new_dir + "ent_embeds.npy")
        rel_embeds = np.load(new_dir + "rel_embeds.npy")
        self.ent_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(ent_embeds))
        self.rel_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(rel_embeds))
        if self.__class__.__name__ == 'TransD':
            ent_transfer = np.load(new_dir + "ent_transfer.npy")
            rel_transfer = np.load(new_dir + "rel_transfer.npy")
            self.ent_transfer = nn.Embedding.from_pretrained(torch.from_numpy(ent_transfer))
            self.rel_transfer = nn.Embedding.from_pretrained(torch.from_numpy(rel_transfer))
        elif self.__class__.__name__ == 'TransH':
            norm_vector = np.load(new_dir + "norm_vector.npy")
            self.norm_vector = nn.Embedding.from_pretrained(torch.from_numpy(norm_vector))
        elif self.__class__.__name__ == 'Analogy':
            norm_vector = np.load(new_dir + "ent_re_embeddings.npy")
            self.ent_re_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(norm_vector))
            norm_vector = np.load(new_dir + "ent_im_embeddings.npy")
            self.ent_im_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(norm_vector))
            norm_vector = np.load(new_dir + "rel_re_embeddings.npy")
            self.rel_re_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(norm_vector))
            norm_vector = np.load(new_dir + "rel_im_embeddings.npy")
            self.rel_im_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(norm_vector))
        elif self.__class__.__name__ == 'TransR':
            norm_vector = np.load(new_dir + "transfer_matrix.npy")
            self.transfer_matrix = nn.Embedding.from_pretrained(torch.from_numpy(norm_vector))
        elif self.__class__.__name__ == 'TransE_ET' or self.__class__.__name__ == 'RESCAL_ET' or self.__class__.__name__ == 'HolE_ET':
            type_embeddings = np.load(new_dir + "type_embeddings.npy")
            self.type_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(type_embeddings))


    def get_score(self, h, r, t):
        return self.calc(h, r, t)

    def trans_score(self, h, r, t):
        score = (h + r) - t
        score = torch.pow(torch.norm(score, 2, -1), 2)
        return score

    def valid(self, stop_metric):

        if len(self.kgs.valid_links) > 0:
            seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities1, self.device)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities2, self.device)), 2, -1)
            '''seed_entity1 = self.ent_embeds(to_tensor(self.kgs.valid_entities1))
            seed_entity2 = self.ent_embeds(to_tensor(self.kgs.valid_entities2))'''

        else:
            seed_entity1 = self.ent_embeds(to_tensor(self.kgs.test_entities1, self.device))
            seed_entity2 = self.ent_embeds(to_tensor(self.kgs.test_entities2, self.device))
        hits1_12, mrr_12 = valid(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(),
                                 None, self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12

    def tests(self, entities1, entities2):
        seed_entity1 = F.normalize(self.ent_embeds(to_tensor(entities1, self.device)), 2, -1)
        seed_entity2 = F.normalize(self.ent_embeds(to_tensor(entities2, self.device)), 2, -1)
        '''seed_entity1 = self.ent_embeds(to_tensor(entities1))
        seed_entity2 = self.ent_embeds(to_tensor(entities2))'''
        _, _, _, sim_list = test(seed_entity1.detach().numpy(), seed_entity2.detach().numpy(),
                                 None,
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm,
                                 csls_k=0, accurate=True)
        print()
        return sim_list

    def load(self):
        dir = self.out_folder.split("/")
        new_dir = ""
        print(dir)
        for i in range(len(dir) - 1):
            new_dir += (dir[i] + "/")
        exist_file = os.listdir(new_dir)
        new_dir = new_dir + "/"
        self.ent_npy = np.load(new_dir + "ent_embeds.npy")
        mapping = None

        print(self.__class__.__name__, type(self.__class__.__name__))
        if self.__class__.__name__ == "GCN_Align":
            print(self.__class__.__name__, "loads attr embeds")
            self.attr_npy = np.load(new_dir + "attr_embeds.npy")

        # if self.__class__.__name__ == "MTransE" or self.__class__.__name__ == "SEA" or self.__class__.__name__ == "KDCoE":
        if os.path.exists(new_dir + "mapping_mat.npy"):
            print(self.__class__.__name__, "loads mapping mat")
            self.map_npy = np.load(new_dir + "mapping_mat.npy")


class parallel_model:
    def __init__(self):
        self.kgs = None
        self.args = None
        self.early_stop = None
        self.flag2 = -1
        self.flag1 = -1
        self.NetworkActor = None

    def init(self, args, kgs):
        self.args = args
        self.kgs = kgs

    def average_weight(self, weights, k):
        average = 0
        for i in range(self.args.parallel_num):
            average += weights[i][k]
        average /= self.args.parallel_num
        return average

    def split_dataset(self):
        kg1_list = task_divide(self.kgs.kg1.relation_triples_list, self.args.parallel_num)
        kg2_list = task_divide(self.kgs.kg2.relation_triples_list, self.args.parallel_num)
        self.NetworkActor[0].set_triple_list.remote(kg1_list[0], kg2_list[0])
        ray.get([self.NetworkActor[i].set_triple_list.remote(kg1_list[i], kg2_list[i]) for i in
                 range(self.args.parallel_num)])

    def test(self):
        ray.get([Actor.test.remote() for Actor in self.NetworkActor])


class parallel_model:
    def __init__(self):
        self.kgs = None
        self.args = None
        self.early_stop = None
        self.flag2 = -1
        self.flag1 = -1
        self.NetworkActor = None
        self.model = None

    def init(self, args, kgs, mod):
        self.args = args
        self.kgs = kgs
        self.model = mod

    def average_weight(self, weights, k):
        average = 0
        for i in range(self.args.parallel_num):
            average += weights[i][k]
        average /= self.args.parallel_num
        return average

    def split_dataset(self):
        kg1_list = task_divide(self.kgs.kg1.relation_triples_list, self.args.parallel_num)
        kg2_list = task_divide(self.kgs.kg2.relation_triples_list, self.args.parallel_num)
        self.NetworkActor[0].set_triple_list.remote(kg1_list[0], kg2_list[0])
        ray.get([self.NetworkActor[i].set_triple_list.remote(kg1_list[i], kg2_list[i]) for i in
                 range(self.args.parallel_num)])

    def test(self):
        ray.get([Actor.test.remote() for Actor in self.NetworkActor])


class align_model_trainer:
    def __init__(self):
        self.kgs = None
        self.args = None
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = None
        self.device = None
        self.model = None
        print(self.device)

    def init(self, args, kgs):
        self.args = args
        self.kgs = kgs

    def valid(self):
        return self.model.valid(self.args.stop_metric)

    def test(self):
        rest_12 = self.model.test(self.kgs.test_entities1, self.kgs.test_entities2)

    def retest(self):
        if self.__class__.__name__ == "gcn_align_trainer":
            self.model = BasicModel(self.args, self.kgs)
            self.model.out_folder = generate_out_folder(self.args.output, self.args.training_data,
                                                        self.args.dataset_division,
                                                        "GCN_Align")
        if self.__class__.__name__ == "rdgcn_trainer":
            self.model = BasicModel(self.args, self.kgs)
            self.model.out_folder = generate_out_folder(self.args.output, self.args.training_data,
                                                        self.args.dataset_division,
                                                        "RDGCN")
        self.model.load()
        t = entity_alignment_evaluation(self.model, self.args, self.kgs)
        t.test()

    def save(self):
        self.model.save()




