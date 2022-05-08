import itertools
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from sklearn import preprocessing
from torch.autograd import Variable

from src.py.evaluation import evaluation
from src.py.load import read
from src.py.load.kg import KG
from src.py.load.kgs import KGs
from src.py.util.env_checker import module_exists


def merge_dic(a, b):
    return {**a, **b}

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


def get_kg_popular_attributes(kg: KG, threshold):
    count_dic = dict()
    for _, attr, _ in kg.attribute_triples_list:
        count_dic[attr] = count_dic.get(attr, 0) + 1
    print("total attributes:", len(count_dic))
    used_attributes_num = int(len(count_dic) * threshold)
    sorted_attributes = sorted(count_dic, key=count_dic.get, reverse=True)
    selected_attributes = set(sorted_attributes[0: used_attributes_num])
    print("selected attributes", len(selected_attributes))
    return selected_attributes


def get_kgs_popular_attributes(kgs: KGs, threshold):
    kg1_selected_attributes = get_kg_popular_attributes(kgs.kg1, threshold)
    kg2_selected_attributes = get_kg_popular_attributes(kgs.kg2, threshold)
    selected_attributes = kg1_selected_attributes | kg2_selected_attributes
    print("total selected attributes", len(selected_attributes))
    return kg1_selected_attributes, kg2_selected_attributes, selected_attributes


def generate_training_data(kgs: KGs, threshold=1.0):
    kg1_selected_attributes, kg2_selected_attributes, selected_attributes = get_kgs_popular_attributes(kgs, threshold)
    entity_attributes_dict = merge_dic(kgs.kg1.entity_attributes_dict, kgs.kg2.entity_attributes_dict)
    print("entity attribute dict", len(entity_attributes_dict))
    training_data_list = list()
    training_links_dict12 = dict(zip(kgs.train_entities1, kgs.train_entities2))
    training_links_dict21 = dict(zip(kgs.train_entities2, kgs.train_entities1))
    training_links_dict = merge_dic(training_links_dict12, training_links_dict21)
    for ent, attributes in entity_attributes_dict.items():
        if ent in training_links_dict.keys():
            attributes = attributes | entity_attributes_dict.get(training_links_dict.get(ent), set())
        attributes = attributes & selected_attributes
        for attr, context_attr in itertools.combinations(attributes, 2):
            if attr != context_attr:
                training_data_list.append((attr, context_attr))
    print("training data of attribute correlations", len(training_data_list))
    return training_data_list


def get_ent_embeds_from_attributes(kgs: KGs, attr_embeds, selected_attributes):
    print("get entity embeddings from attributes")
    start = time.time()
    ent_mat = None
    entity_attributes_dict = merge_dic(kgs.kg1.entity_attributes_dict, kgs.kg2.entity_attributes_dict)
    zero_vec = np.zeros([1, attr_embeds.shape[1]], dtype=np.float32)
    for i in range(kgs.entities_num):
        attr_vec = zero_vec
        attributes = entity_attributes_dict.get(i, set())
        attributes = attributes & selected_attributes
        if len(attributes) > 0:
            attr_vecs = attr_embeds[list(attributes), ]
            attr_vec = np.mean(attr_vecs, axis=0, keepdims=True)
        if ent_mat is None:
            ent_mat = attr_vec
        else:
            ent_mat = np.row_stack([ent_mat, attr_vec])
    print('cost time: {:.4f}s'.format(time.time() - start))
    return ent_mat
    # return F.normalize(ent_mat)


class Attr2Vec(nn.Module):
    def set_kgs(self, kgs):
        self.kgs = kgs
        _, _, self.selected_attributes = get_kgs_popular_attributes(kgs, 0.9)
        self.num_sampled_negs = len(self.selected_attributes) // 5

    def set_args(self, args):
        self.args = args
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division,
                            self.__class__.__name__)
        # self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division, self.__class__.__name__)

    def init(self):
        self._define_variables()

    def __init__(self):
        super(Attr2Vec, self).__init__()
        self.train_labels = None
        self.train_inputs_embed = None
        self.train_inputs = None
        self.num_sampled_negs = -1
        self.kgs = None
        self.args = None
        self.out_folder = None
        self.flag1, self.flag2 = -1, -1
        self.early_stop = False
        self.session = None
        self.selected_attributes = None
        self.opt = 'Adagrad'

    def _define_variables(self):
        initrange = 0.5 / self.args.dim
        self.embeds = nn.Embedding(self.kgs.attributes_num, self.args.dim)
        self.criterion = nn.LogSigmoid()
        print("总属性数目是：{}".format(self.kgs.attributes_num))
        self.nce_weights = nn.Embedding(self.kgs.attributes_num, self.args.dim)
        # self.embeds.weight.data.uniform_(-initrange, initrange)
        nn.init.xavier_uniform_(self.embeds.weight.data)
        nn.init.xavier_uniform_(self.nce_weights.weight.data)
        self.nce_biases = Variable(torch.zeros(self.kgs.attributes_num), requires_grad=True)

    def define_embed_graph(self, data):
        pos_attr1 = data['pos_1']
        pos_attr2 = data['pos_2']
        neg_attr2 = data['neg_2']
        # print(pos_attr1)
        batch_size = pos_attr1.shape[0]
        pos_attr1 = pos_attr1.view(batch_size, -1)
        pos_attr2 = pos_attr2.view(batch_size, -1)
        # neg_attr1 = neg_attr1.view(batch_size, -1, 1)
        neg_attr2 = neg_attr2.view(batch_size, -1)
        # pos1 = F.normalize(self.embeds(pos_attr1), 2, -1)
        # pos2 = F.normalize(self.embeds(pos_attr2), 2, -1).permute(0, 2, 1)
        pos1 = self.embeds(pos_attr1)
        pos2 = self.nce_weights(pos_attr2).permute(0, 2, 1)
        neg2 = self.nce_weights(neg_attr2).permute(0, 2, 1)
        pos = torch.bmm(pos1, pos2).squeeze(2).flatten()
        neg = torch.bmm(pos1, neg2.neg()).squeeze(1).flatten()
        return -(self.criterion(pos).mean() + self.criterion(neg).mean()) / 2
        # input_embedding = pos1.unsqueeze(2)  # [batch_size, embed_size, 1],新增一个维度用于向量乘法
        # input_embedding = input_embedding.view(BATCH_SIZE, EMBEDDING_DIM, 1)
        # neg1 = F.normalize(self.embeds(neg_attr1), 2, -1)
        # neg2 = F.normalize(self.embeds(neg_attr2), 2, -1).permute(0, 2, 1)
        # pos = torch.matmul(pos1, pos2).flatten()
        # neg = torch.matmul(pos1, neg2).flatten()
        # return -(self.criterion(pos).mean() + self.criterion(-neg).mean()) / 2'''

    def eval_attribute_embeddings(self):
        # return np.load(r'D:/OPENEA-pytorch/output/attr_embeds.npy')
        try:
            dir = self.out_folder.split("/")
            new_dir = ""
            print(dir)
            for i in range(len(dir) - 1):
                new_dir += (dir[i] + "/")
            new_dir = new_dir + "/"
            embeds = np.load(new_dir + "attr_embeds.npy")
            return embeds
        except:
            a = self.embeds.weight.data
            return a.cpu().detach().numpy()

    def eval_kg1_ent_embeddings(self):
        mat = get_ent_embeds_from_attributes(self.kgs, self.eval_attribute_embeddings(), self.selected_attributes)
        embeds1 = mat[self.kgs.kg1.entities_list,]
        return embeds1

    def eval_kg2_ent_embeddings(self):
        mat = get_ent_embeds_from_attributes(self.kgs, self.eval_attribute_embeddings(), self.selected_attributes)
        embeds2 = mat[self.kgs.kg2.entities_list,]
        return embeds2

    def eval_sim_mat(self):
        mat = get_ent_embeds_from_attributes(self.kgs, self.eval_attribute_embeddings(), self.selected_attributes)
        embeds1 = mat[self.kgs.valid_entities1 + self.kgs.test_entities1,]
        embeds2 = mat[self.kgs.valid_entities2 + self.kgs.test_entities2,]
        return np.matmul(embeds1, embeds2.T)

    def valid(self):
        mat = get_ent_embeds_from_attributes(self.kgs, self.eval_attribute_embeddings(), self.selected_attributes)
        embeds1 = mat[self.kgs.valid_entities1, ]
        embeds2 = mat[self.kgs.valid_entities2, ]
        hits1_12, mrr_12 = evaluation.valid(embeds1, embeds2, None, self.args.top_k,
                                            self.args.test_threads_num, metric=self.args.eval_metric)
        if self.args.stop_metric == 'hits1':
            return hits1_12
        return mrr_12

    def test(self, save=True):
        mat = get_ent_embeds_from_attributes(self.kgs, self.eval_attribute_embeddings(), self.selected_attributes)
        embeds1 = mat[self.kgs.test_entities1, ]
        embeds2 = mat[self.kgs.test_entities2, ]
        rest_12, _, _, rest_21, _, _ = evaluation.test(embeds1, embeds2, None, self.args.top_k,
                                                       self.args.test_threads_num, metric=self.args.eval_metric,
                                                       csls_k=self.args.csls)
        if save:
            ent_ids_rest_12 = [(self.kgs.test_entities1[i], self.kgs.test_entities2[j]) for i, j in rest_12]
            #read.save_results(self.out_folder, ent_ids_rest_12)

    def save(self):
        out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division,
                                              self.__class__.__name__)
        embeds = self.embeds.cpu().weight.data
        read.save_embeddings(out_folder, self.kgs, None, None, embeds)
