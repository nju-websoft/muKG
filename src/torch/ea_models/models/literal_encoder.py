import time

import numpy as np
import gc
# from sklearn import preprocessing
from gensim.models.word2vec import Word2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trainer.util import get_optimizer, to_tensor


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s


class AutoEncoderModel(nn.Module):
    def __init__(self, word_vec_list, args, hidden_dimensions=None):
        super(AutoEncoderModel, self).__init__()
        self.args = args
        self.weights, self.biases = nn.ParameterDict(), nn.ParameterDict()
        self.input_dimension = self.args.literal_len * self.args.word2vec_dim
        if hidden_dimensions is None:
            hidden_dimensions = [1024, 512, self.args.dim]
        self.hidden_dimensions = hidden_dimensions
        self.layer_num = len(self.hidden_dimensions)
        self.encoder_output = None
        self.decoder_output = None
        self.decoder_op = None

        self.word_vec_list = to_tensor(np.reshape(word_vec_list, [len(word_vec_list), self.input_dimension]))
        if self.args.encoder_normalize:
            self.word_vec_list = F.normalize(self.word_vec_list, 2, -1)

        self._init_graph()

    def _init_graph(self):
        self.hidden_dimensions.insert(0, self.input_dimension)
        hds = self.hidden_dimensions
        for i in range(self.layer_num):
            tmp = torch.randn(hds[i], hds[i + 1])
            self.weights.update({'encoder_h' + str(i): nn.Parameter(tmp)})
            bias = torch.randn(hds[i + 1])
            self.biases.update({'encoder_b' + str(i): nn.Parameter(bias)})
        for i in range(self.layer_num):
            i_decoder = self.layer_num - i
            tmp = torch.randn(hds[i_decoder], hds[i_decoder - 1])
            self.weights.update({'decoder_h' + str(i): nn.Parameter(tmp)})
            bias = torch.randn(hds[i_decoder - 1])
            self.biases.update({'decoder_b' + str(i): nn.Parameter(bias)})

    def forward(self, data):
        encoder_output = self.encoder(data)
        if self.args.encoder_normalize:
            encoder_output = F.normalize(encoder_output, 2, -1)
        decoder_output = self.decoder(encoder_output)
        return torch.mean(torch.pow(decoder_output - data, 2))
        # self.optimizer = generate_optimizer(self.loss, self.args.learning_rate, opt=self.args.optimizer)

    def encoder(self, input_data):
        input_layer = input_data
        for i in range(self.layer_num):
            input_layer = torch.matmul(input_layer, self.weights['encoder_h' + str(i)]) + self.biases['encoder_b' + str(i)]
            if self.args.encoder_active == 'sigmoid':
                input_layer = torch.sigmoid(input_layer)
            elif self.args.encoder_active == 'tanh':
                input_layer = torch.tanh(input_layer)
        encoder_output = input_layer
        return encoder_output

    def decoder(self, input_data):
        input_layer = input_data
        for i in range(self.layer_num):
            input_layer = torch.matmul(input_layer, self.weights['decoder_h' + str(i)]) + self.biases['decoder_b' + str(i)]
            if self.args.encoder_active == 'sigmoid':
                input_layer = torch.sigmoid(input_layer)
            elif self.args.encoder_active == 'tanh':
                input_layer = torch.tanh(input_layer)
        decoder_output = input_layer
        return decoder_output

    def encoder_multi_batches(self, input_data):
        print('encode literal embeddings...', len(input_data))
        batches = list()
        results = np.zeros((len(input_data), self.args.dim))
        batch_size = self.args.batch_size
        num_batch = len(input_data) // batch_size + 1
        for i in range(num_batch):
            if i == num_batch - 1:
                batches.append(input_data[i * batch_size:])
            else:
                batches.append(input_data[i * batch_size:(i + 1) * batch_size])

        for batch_i in range(num_batch):
            input_layer = np.reshape(batches[batch_i], [len(batches[batch_i]), self.input_dimension])
            for i in range(self.layer_num):
                weight_i = self.weights['encoder_h' + str(i)].detach().numpy()
                bias_i = self.biases['encoder_b' + str(i)].detach().numpy()
                input_layer = np.matmul(input_layer, weight_i) + bias_i
                if self.args.encoder_active == 'sigmoid':
                    input_layer = sigmoid(input_layer)
                elif self.args.encoder_active == 'tanh':
                    input_layer = tanh(input_layer)
            literal_vectors = input_layer
            if batch_i == num_batch - 1:
                results[batch_i * batch_size:] = np.array(literal_vectors)
            else:
                results[batch_i * batch_size:(batch_i + 1) * batch_size] = np.array(literal_vectors)
            del literal_vectors
            gc.collect()
        print("encoded literal embeddings", results.shape)
        np.save('../../literal_embeddings.npy', results)
        return results



def generate_unlisted_word2vec(word2vec, literal_list, vector_dimension):
    unlisted_words = []
    for literal in literal_list:
        words = literal.split(' ')
        for word in words:
            if word not in word2vec:
                unlisted_words.append(word)

    character_vectors = {}
    alphabet = ''
    ch_num = {}
    for word in unlisted_words:
        for ch in word:
            n = 1
            if ch in ch_num:
                n += ch_num[ch]
            ch_num[ch] = n
    ch_num = sorted(ch_num.items(), key=lambda x: x[1], reverse=True)
    ch_sum = sum([n for (_, n) in ch_num])
    for i in range(len(ch_num)):
        if ch_num[i][1] / ch_sum >= 0.0001:
            alphabet += ch_num[i][0]
    print(alphabet)
    print('len(alphabet):', len(alphabet), '\n')
    char_sequences = [list(word) for word in unlisted_words]
    model = Word2Vec(char_sequences, size=vector_dimension, window=5, min_count=1)
    for ch in alphabet:
        assert ch in model.wv
        character_vectors[ch] = model.wv[ch]

    word2vec_new = {}
    for word in unlisted_words:
        vec = np.zeros(vector_dimension, dtype=np.float32)
        for ch in word:
            if ch in alphabet:
                vec += character_vectors[ch]
        if len(word) != 0:
            word2vec_new[word] = vec / len(word)

    word2vec.update(word2vec_new)
    return word2vec


class LiteralEncoder:

    def __init__(self, literal_list, word2vec, args, word2vec_dimension):
        self.args = args
        self.literal_list = literal_list
        self.word2vec = generate_unlisted_word2vec(word2vec, literal_list, word2vec_dimension)
        self.tokens_max_len = self.args.literal_len
        self.word2vec_dimension = word2vec_dimension
        literal_vector_list = []
        for literal in self.literal_list:
            vectors = np.zeros((self.tokens_max_len, self.word2vec_dimension), dtype=np.float32)
            words = literal.split(' ')
            for i in range(min(self.tokens_max_len, len(words))):
                if words[i] in self.word2vec:
                    vectors[i] = self.word2vec[words[i]]
            literal_vector_list.append(vectors)
        assert len(literal_list) == len(literal_vector_list)
        self.encoder_model = AutoEncoderModel(literal_vector_list, self.args)
        self.input_dimension = self.args.literal_len * self.args.word2vec_dim
        self.word_vec_list = np.reshape(literal_vector_list, [len(literal_vector_list), self.input_dimension])
        self.optimizer = get_optimizer('Adagrad', self.encoder_model, 0.01)
        for i in range(self.args.encoder_epoch):
            self.train_one_epoch(i + 1)
        self.encoded_literal_vector = self.encoder_model.encoder_multi_batches(literal_vector_list)

    def train_one_epoch(self, epoch):
        start_time = time.time()

        batches = list()
        batch_size = self.args.batch_size
        num_batch = len(self.word_vec_list) // batch_size + 1
        for i in range(num_batch):
            if i == num_batch - 1:
                batches.append(self.word_vec_list[i * batch_size:])
            else:
                batches.append(self.word_vec_list[i * batch_size:(i + 1) * batch_size])

        loss_sum = 0.0
        for batch_i in range(num_batch):
            self.optimizer.zero_grad()
            loss_train = self.encoder_model(to_tensor(batches[batch_i]))
            loss_train.backward()
            self.optimizer.step()
            loss_sum += loss_train
        loss_sum /= self.args.batch_size
        end = time.time()
        print('epoch {} of literal encoder, loss: {:.4f}, time: {:.4f}s'.format(epoch, loss_sum, end - start_time))
        return


