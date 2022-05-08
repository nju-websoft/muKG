import math
import multiprocessing as mp
import random
import sys
import time
import gc
import numpy as np
import os

import tensorflow as tf
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


class BasicModel(tf.keras.Model):
    def init(self):
        # need to be overwrite
        pass

    def __init__(self, args, kgs):
        super(BasicModel, self).__init__()
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

    def eval_valid(self):
        ent_embeds = self.ent_embeddings
        rel_embeds = self.rel_embeddings
        return ent_embeds, rel_embeds

    def save(self):
        ent_embeds = self.ent_embeddings.numpy()
        rel_embeds = self.rel_embeddings.numpy()
        read.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, None)

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
            self.ent_re_embeddings.assign(ent_re_embeddings)
            self.ent_im_embeddings.assign(ent_im_embeddings)
            self.rel_re_embeddings.assign(rel_re_embeddings)
            self.rel_im_embeddings.assign(rel_im_embeddings)
            return
        ent_embeds = np.load(new_dir + "ent_embeds.npy")
        rel_embeds = np.load(new_dir + "rel_embeds.npy")
        self.ent_embeddings.assign(ent_embeds)
        self.rel_embeddings.assign(rel_embeds)
        if self.__class__.__name__ == 'TransD':
            ent_transfer = np.load(new_dir + "ent_transfer.npy")
            rel_transfer = np.load(new_dir + "rel_transfer.npy")
            self.ent_transfer.assign(ent_transfer)
            self.rel_transfer.assign(rel_transfer)
        elif self.__class__.__name__ == 'TransH':
            norm_vector = np.load(new_dir + "norm_vector.npy")
            self.norm_vector.assign(norm_vector)
        elif self.__class__.__name__ == 'Analogy':
            norm_vector = np.load(new_dir + "ent_re_embeddings.npy")
            self.ent_re_embeddings.assign(norm_vector)
            norm_vector = np.load(new_dir + "ent_im_embeddings.npy")
            self.ent_im_embeddings.assign(norm_vector)
            norm_vector = np.load(new_dir + "rel_re_embeddings.npy")
            self.rel_re_embeddings.assign(norm_vector)
            norm_vector = np.load(new_dir + "rel_im_embeddings.npy")
            self.rel_im_embeddings.assign(norm_vector)
        elif self.__class__.__name__ == 'TransR':
            norm_vector = np.load(new_dir + "transfer_matrix.npy")
            self.transfer_matrix.assign(norm_vector)

    def get_pos_score(self, score):
        tmp = score[:self.batch_size]
        return tf.reshape(tmp, [self.batch_size, -1])

    def get_neg_score(self, score):
        tmp = score[self.batch_size:]
        return tf.reshape(tmp, [self.batch_size, -1])

    def get_score(self, h, r, t):
        return self.calc(h, r, t)



