import itertools
import math
import random
import time
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from src.py.base.optimizers import get_optimizer_torch
from src.py.load.kg import KG
from src.py.load.kgs import KGs
from src.py.util.util import to_tensor_cpu, to_tensor
from src.torch.ea_models.models.attr2vec import Attr2Vec


def early_stop(flag1, flag2, flag):
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False


def merge_dic(a, b):
    return {**a, **b}


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


class attr2vec_trainer:
    def __init__(self, args, kgs):
        self.args = args
        self.kgs = kgs
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = None
        self.optimizer = None
        self.model = Attr2Vec()
        if self.args.is_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.model.set_args(args)
        self.model.set_kgs(kgs)
        self.model.init()
        self.model.to(self.device)
         
    def launch_training_1epo(self, epoch, dataloader, steps, training_data_list):
        start = time.time()
        epoch_loss = 0
        trained_pos_triples = len(training_data_list)
        self.optimizer = get_optimizer_torch(self.args.optimizer, self.model, self.args.learning_rate)
        for data in dataloader:
            self.optimizer.zero_grad()
            batch_loss = self.model.define_embed_graph({'pos_1' : to_tensor(data['pos_1'], self.device),
                                                        'pos_2' : to_tensor(data['pos_2'], self.device),
                                                        'neg_2': to_tensor(data['neg_2'], self.device)})
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss
        print('epoch {}, attribute loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def test(self):
        #rest_12 = self.model.test(self.kgs.test_entities1, self.kgs.test_entities2)
        pass

    def run(self):
        t = time.time()
        training_data_list = generate_training_data(self.kgs, threshold=0.9)
        data_loader = LoadAttrDataset(self.kgs, training_data_list, self.args.batch_size,
                                      8, 10)
        steps = len(training_data_list) // self.args.batch_size
        for i in range(1, self.args.attr_max_epoch + 1):
        #for i in range(3):
            self.launch_training_1epo(i, data_loader, steps, training_data_list)
            if i % 50 == 0:
                self.model.valid()
        self.model.save()
        # np.save(r'D:/OPENEA-pytorch/output/attr_embeds.npy', self.model.eval_attribute_embeddings())

    def get_sim_mat(self):
        return self.model.eval_sim_mat()


class LoadAttrDataset(DataLoader):
    def __init__(self, kgs, training_data_list, batch_size, threads, neg_size):
        self.batch_size = batch_size
        self.kgs = kgs
        self.training_data_list = training_data_list
        self.neg_size = neg_size
        self.data = self.__construct_dataset()
        super(LoadAttrDataset, self).__init__(
            dataset=self.data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=threads,
            pin_memory=True,
            collate_fn=self.data.collate_fn,
            drop_last=False
        )

    def __construct_dataset(self):
        train_dataset = TrainDataset(self.training_data_list, self.kgs.attributes_num,
                                     self.batch_size, self.neg_size)
        return train_dataset

    def get_ent_tot(self):
        return self.data.get_ent_tot()

    def get_rel_tot(self):
        return self.data.get_rel_tot()

    def get_batch_size(self):
        return self.batch_size


class TrainDataset(Dataset):
    def __init__(self, data, attr_num, batch_size, neg_size):  # add parameters here
        self.data = data
        self.neg_size = neg_size
        self.batch_size = batch_size
        self.attr_num = attr_num
        # self.sampling_array, self.class_probs = make_sampling_array(attr_num, None)
        # self.__count_htr()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_neg(self):
        result = []
        num = list(range(self.attr_num))
        result = random.sample(num, self.neg_size)
        return result

    def sample_values(self, true_classes, num_sampled, unique, no_accidental_hits, sampling_array, class_probs):
        """ Samples negative items for the calculation of the NCE loss. Operates on batches of targets. """
        # Initialize output sequences
        sampled_candidates = np.zeros(num_sampled)
        true_expected_count = np.zeros(true_classes.size())
        sampled_expected_count = np.zeros(num_sampled)

        # If the true labels should not be sampled as a noise items, add them all to the rejected list
        if no_accidental_hits:
            rejected = list()
        else:
            rejected = true_classes.tolist()
        # Assign true label probabilities
        rows, cols = true_classes.size()
        for i in range(rows):
            true_expected_count[i][0] = class_probs[true_classes.numpy()[i][0]]
        # Obtain sampled items and their probabilities
        print('Sampling items and their probabilities.')
        for k in range(num_sampled):
            sampled_pos = np.random.randint(int(1e8))
            sampled_idx = sampling_array[sampled_pos]
            if unique:
                while sampled_idx in rejected:
                    sampled_idx = sampling_array[np.random.randint(0, int(1e8))]
            # Append sampled candidate and its probability to the output sequences for current target
            sampled_candidates[k] = sampled_idx
            sampled_expected_count[k] = class_probs[sampled_idx]
            # Re-normalize probabilities
            if unique:
                class_probs = renormalize(class_probs, sampled_idx)

        # Process outputs before they are returned
        sampled_candidates = sampled_candidates.astype(np.int64, copy=False)
        true_expected_count = true_expected_count.astype(np.float32, copy=False)
        sampled_expected_count = sampled_expected_count.astype(np.float32, copy=False)

        return Variable(torch.LongTensor(sampled_candidates)), \
               Variable(torch.FloatTensor(true_expected_count)), \
               Variable(torch.FloatTensor(sampled_expected_count))

    def collate_fn(self, data):  # this fuc is used to deal with each batch from dataloader
        batch_data = {}
        batch_size = len(data)
        pos1 = [x[0] for x in data]
        pos2 = [x[1] for x in data]
        # 接下来进行负样本选择，记住选择那些h 和 t没有什么关系的数据来添加进入数据中
        neg_batch = []
        for i in range(batch_size):
            neg = self.get_neg()
            neg_batch += neg
        # neg_h = [h[0] for h in neg_batch]
        batch_data['pos_1'] = pos1
        batch_data['pos_2'] = pos2
        batch_data['neg_2'] = neg_batch
        # batch_data['neg_1'] = to_var(neg_h)
        '''batch_data['neg_2'] = self.sample_values(batch_data['pos_2'].view(batch_size, -1), batch_size*10, True,
                                                 True, self.sampling_array,
                                                 self.class_probs)'''
        return batch_data

    def get_ent_tot(self):
        return self.head_total

    def get_rel_tot(self):
        return self.rel_total

    def get_neg_size(self):
        return self.neg_size