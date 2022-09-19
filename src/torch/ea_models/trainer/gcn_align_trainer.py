import gc
import math
import random
import time
import torch
import numpy as np
from torch.autograd import Variable

from src.py.base.optimizers import get_optimizer_torch
from src.py.evaluation.evaluation import valid, test
from src.py.load import read
from src.py.util.util import to_var, generate_out_folder, to_tensor_cpu, to_tensor
from src.torch.ea_models.models.gcn_align import GCN_Utils, load_attr, GCN_Align_Unit
from src.torch.kge_models.basic_model import align_model_trainer


def early_stop(flag1, flag2, flag):
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False


class gcn_align_trainer(align_model_trainer):
    def __init__(self):
        super(gcn_align_trainer, self).__init__()
        self.out_folder = None
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = None
        self.optimizer1 = None
        self.optimizer2 = None
        self.attr = None
        self.opt = 'Adam'
        self.act_func = torch.relu
        self.dropout = 0.0
        # *****************************add*******************************************************
        self.struct_loss = None
        self.struct_optimizer = None
        self.vec_ae = None
        self.vec_se = None
        self.num_supports = None
        self.utils = None
        self.adj = None
        self.ae_input = None
        self.train = None
        self.e = None
        self.support = None
        self.adj = None
        self.ph_ae = None
        self.ph_se = None
        self.model_ae = None
        self.model_se = None
        self.feed_dict_se = None
        self.feed_dict_ae = None

    def init(self, args, kgs):
        self.args = args
        self.kgs = kgs
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division,
                                              'GCN_Align')
        assert self.args.alignment_module == 'mapping'
        assert self.args.neg_triple_num > 1
        assert self.args.learning_rate >= 0.01
        self.num_supports = self.args.support_number
        self.utils = GCN_Utils(self.args, self.kgs)
        self.attr = load_attr(self.kgs.entities_num, self.kgs)
        self.adj, self.ae_input, self.train = self.utils.load_data(self.attr)
        self.e = self.ae_input[2][0]
        self.support = [self.utils.preprocess_adj(self.adj)]
        '''self.ph_ae = {
            "support": [tf.sparse_placeholder(tf.float32) for _ in range(self.args.support_number)],
            "features": tf.sparse_placeholder(tf.float32),
            "dropout": tf.placeholder_with_default(0., shape=()),
            "num_features_nonzero": tf.placeholder_with_default(0, shape=())
        }
        self.ph_se = {
            "support": [tf.sparse_placeholder(tf.float32) for _ in range(self.args.support_number)],
            "features": tf.placeholder(tf.float32),
            "dropout": tf.placeholder_with_default(0., shape=()),
            "num_features_nonzero": tf.placeholder_with_default(0, shape=())
        }'''
        if self.args.is_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.model_ae = GCN_Align_Unit(self.args, self.support, input_dim=self.ae_input[2][1],
                                       output_dim=self.args.ae_dim,
                                       sparse_inputs=True, featureless=False, logging=False)
        self.model_se = GCN_Align_Unit(self.args, self.support, input_dim=self.e, output_dim=self.args.se_dim,
                                       sparse_inputs=False,
                                       featureless=True, logging=False)
        print(self.model_ae.parameters())
        self.model_se.to(self.device)
        self.model_ae.to(self.device)
        self.optimizer1 = get_optimizer_torch('SGD', self.model_ae, self.args.learning_rate)
        self.optimizer2 = get_optimizer_torch('SGD', self.model_se, self.args.learning_rate)

    def train_embeddings(self):
        # **t=train_number k=neg_num
        neg_num = self.args.neg_triple_num
        train_num = len(self.kgs.train_links)
        train_links = np.array(self.kgs.train_links)
        pos = np.ones((train_num, neg_num)) * (train_links[:, 0].reshape((train_num, 1)))
        neg_left = pos.reshape((train_num * neg_num,))
        pos = np.ones((train_num, neg_num)) * (train_links[:, 1].reshape((train_num, 1)))
        neg2_right = pos.reshape((train_num * neg_num,))
        neg2_left = None
        neg_right = None
        feed_dict_se = None
        feed_dict_ae = None
        indices = []
        x_in = [x[0] for x in self.ae_input[0]]
        y_in = [y[1] for y in self.ae_input[0]]
        indices.append(x_in)
        indices.append(y_in)
        features = torch.sparse_coo_tensor(indices=to_tensor_cpu(indices), values=self.ae_input[1],
                                           size=self.ae_input[2]).to(self.device)
        for i in range(1, self.args.max_epoch + 1):
            start = time.time()
            if i % 10 == 1:
                neg2_left = np.random.choice(self.e, train_num * neg_num)
                neg_right = np.random.choice(self.e, train_num * neg_num)
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            batch_loss1 = self.model_ae({
                'neg_left': to_var(neg_left, self.device), 'neg_right': to_var(neg_right, self.device),
                'neg2_left': to_var(neg2_left, self.device), 'neg2_right': to_var(neg2_right, self.device),
                'features': features, 'support': self.support[0].to(torch.float32).to(self.device),
                'ILL0': to_var(self.train[:, 0], self.device), 'ILL1': to_var(self.train[:, 1], self.device)
            })
            batch_loss2 = self.model_se({
                'neg_left': to_var(neg_left, self.device), 'neg_right': to_var(neg_right, self.device),
                'neg2_left': to_var(neg2_left, self.device), 'neg2_right': to_var(neg2_right, self.device),
                'features': to_tensor(1., self.device), 'support': self.support[0].to(torch.float32).to(self.device),
                'ILL0': to_var(self.train[:, 0], self.device), 'ILL1': to_var(self.train[:, 1], self.device)
            })
            batch_loss1.backward()
            batch_loss2.backward()
            self.optimizer1.step()
            self.optimizer2.step()
            gc.collect()
            batch_loss = batch_loss1 + batch_loss2
            print('epoch {}, avg. relation triple loss: {:.4f}, cost time: {:.4f}s'.format(i, batch_loss,
                                                                                           time.time() - start))
            # ********************no early stop********************************************
            if i >= 10 and i % self.args.eval_freq == 0:
                self.feed_dict_se = feed_dict_se
                self.feed_dict_ae = feed_dict_ae
                flag = self.valid_(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.args.no_early:
                    self.early_stop = False
                if self.early_stop or i == self.args.max_epoch:
                    break

    def test(self, save=True):
        self.vec_se = self.model_se.get_output()
        self.vec_ae = self.model_ae.get_output()
        if self.args.test_method == "sa":
            beta = self.args.beta
            embeddings = np.concatenate([self.vec_se * beta, self.vec_ae * (1.0 - beta)], axis=1)
        else:
            embeddings = self.vec_se
        embeds1 = np.array([embeddings[e] for e in self.kgs.test_entities1])
        embeds2 = np.array([embeddings[e] for e in self.kgs.test_entities2])
        rest_12, _, _, _ = test(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
                                metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        test(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls, accurate=True)
        '''if save:
            ent_ids_rest_12 = [(self.kgs.test_entities1[i], self.kgs.test_entities2[j]) for i, j in rest_12]
            rd.save_results(self.out_folder, ent_ids_rest_12)'''

    def save(self):
        ent_embeds = self.vec_se
        attr_embeds = self.vec_ae
        read.save_embeddings(self.out_folder, self.kgs, ent_embeds, None, attr_embeds, mapping_mat=None)

    def valid_(self, stop_metric):
        se = self.model_se.get_output()
        if self.args.test_method == "sa":
            ae = self.model_ae.get_output()
            beta = self.args.beta
            embeddings = np.concatenate([se * beta, ae * (1.0 - beta)], axis=1)
        else:
            embeddings = se
        embeds1 = np.array([embeddings[e] for e in self.kgs.valid_entities1])
        embeds2 = np.array([embeddings[e] for e in self.kgs.valid_entities2 + self.kgs.test_entities2])
        hits1_12, mrr_12 = valid(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
                                 metric=self.args.eval_metric)
        if stop_metric == 'hits1':
            return hits1_12
        return mrr_12

    def run(self):
        t = time.time()
        self.train_embeddings()
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
        self.test(False)
        self.save()
