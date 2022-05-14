import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.py.base.optimizers import get_optimizer_torch
from src.py.evaluation.evaluation import test, valid
from src.py.load import read
from src.py.util.util import early_stop, to_var
from src.torch.ea_models.models.rdgcn import Layer, get_neg
from src.torch.kge_models.basic_model import align_model_trainer


def generate_out_folder(out_folder, training_data_path, div_path, method_name):
    params = training_data_path.strip('/').split('/')
    print(out_folder, training_data_path, params, div_path, method_name)
    path = params[-1]
    folder = out_folder + method_name + '/' + path + "/" + div_path + "/"
    print("results output folder:", folder)
    return folder
    
    
class rdgcn_trainer(align_model_trainer):
    def __init__(self):
        super(rdgcn_trainer, self).__init__()
        self.early_stop = None
        self.flag2 = -1
        self.flag1 = -1
        self.loss = 0
        self.output = None
        self.optimizer = None
        self.model_init = None
        self.sess = None
        self.feeddict = None
        self.gcn_model = None
        self.local_name_vectors = None
        self.entity_local_name_dict = None
        self.entities = None
        self.word_embed = '../../datasets/wiki-news-300d-1M.vec'

    def init(self, args, kgs):
        self.args = args
        self.kgs = kgs
        '''if self.args.is_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:'''
        self.device = torch.device('cpu')
        self.entities = self.kgs.kg1.entities_set | self.kgs.kg2.entities_set
        self.gcn_model = Layer(self.args, self.kgs, self.local_name_vectors)
        self.gcn_model.to(self.device)
        self.gcn_model.init()
        # self.output, self.loss = self.gcn_model.build()
        self.optimizer = get_optimizer_torch('Adam', self.gcn_model, self.args.learning_rate)

    def training1(self):
        neg_num = self.args.neg_triple_num
        train_num = len(self.kgs.train_links)
        train_links = np.array(self.kgs.train_links)
        pos = np.ones((train_num, neg_num)) * (train_links[:, 0].reshape((train_num, 1)))
        neg_left = pos.reshape((train_num * neg_num,))
        pos = np.ones((train_num, neg_num)) * (train_links[:, 1].reshape((train_num, 1)))
        neg2_right = pos.reshape((train_num * neg_num,))
        # output = self.sess.run(self.output)
        # neg2_left = get_neg(train_links[:, 1], output, self.args.neg_triple_num)
        # neg_right = get_neg(train_links[:, 0], output, self.args.neg_triple_num)
        # self.feeddict = {"neg_left:0": neg_left,
        #                  "neg_right:0": neg_right,
        #                  "neg2_left:0": neg2_left,
        #                  "neg2_right:0": neg2_right}
        for i in range(1, self.args.max_epoch + 1):
            start = time.time()
            if i % 10 == 1:
                output = self.gcn_model.get_output()
                neg2_left = get_neg(train_links[:, 1], output, self.args.neg_triple_num)
                neg_right = get_neg(train_links[:, 0], output, self.args.neg_triple_num)
            self.optimizer.zero_grad()
            batch_loss = self.gcn_model.build({"neg_left": to_var(neg_left, self.device),
                                               "neg_right": to_var(neg_right, self.device),
                                               "neg2_left": to_var(neg2_left, self.device),
                                               "neg2_right": to_var(neg2_right, self.device)})
            batch_loss.backward()
            self.optimizer.step()
            print('epoch {}, avg. relation triple loss: {:.4f}, cost time: {:.4f}s'.format(i, batch_loss,
                                                                                           time.time() - start))

            # ********************no early stop********************************************
            if i >= 10 and i % self.args.eval_freq == 0:
                flag = self.valid_(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.args.no_early:
                    self.early_stop = False
                if self.early_stop or i == self.args.max_epoch:
                    break
        self.test()
        self.save()

    def test(self, save=True):
        embedding = self.gcn_model.get_output()
        embeds1 = np.array([embedding[e] for e in self.kgs.test_entities1])
        embeds2 = np.array([embedding[e] for e in self.kgs.test_entities2])
        rest_12 = test(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
                             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        test(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls, accurate=True)
        '''if save:
            rd.save_results(self.out_folder, rest_12)'''

    def save(self):
        out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division,
                                              self.__class__.__name__)
        embedding = self.gcn_model.get_output()
        read.save_embeddings(out_folder, self.kgs, embedding, None, None, mapping_mat=None)

    def valid_(self, stop_metric):
        embedding = self.gcn_model.get_output()
        embeds1 = np.array([embedding[e] for e in self.kgs.valid_entities1])
        embeds2 = np.array([embedding[e] for e in self.kgs.valid_entities2])
        hits1_12, mrr_12 = valid(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
                                 metric=self.args.eval_metric)
        if stop_metric == 'hits1':
            return hits1_12
        return mrr_12

    def run(self):
        t = time.time()
        self.training1()
        print("training finish")
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))