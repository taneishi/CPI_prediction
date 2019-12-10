import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from sklearn.metrics import roc_auc_score, precision_score, recall_score

import timeit
import sys

class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self, n_fingerprint, n_word, dim, window, layer_gnn, layer_cnn, layer_output):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.layer_gnn = layer_gnn
        self.layer_cnn = layer_cnn
        self.layer_output = layer_output
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim, self.layer_gnn)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=1, 
            kernel_size=2*window+1, stride=1, padding=window) for _ in range(self.layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim) for _ in range(self.layer_output)])
        self.W_interaction = nn.Linear(2*dim, 2)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        '''The attention mechanism is applied to the last layer of CNN.'''

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words = inputs

        '''Compound vector with GNN.'''
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, self.layer_gnn)

        '''Protein vector with attention-CNN.'''
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector, word_vectors, self.layer_cnn)

        '''Concatenate the above two vectors and output the interaction.'''
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(self.layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)

        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores

def train(model, dataset):
    model.train()
    N = len(dataset)
    loss_total = 0
    for data in dataset:
        loss = model(data)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        loss_total += loss.to('cpu').data.numpy()
    return loss_total

def test(model, dataset):
    model.eval()
    N = len(dataset)
    T, Y, S = [], [], []
    for data in dataset:
        (correct_labels, predicted_labels, predicted_scores) = model(data, train=False)
        T.append(correct_labels)
        Y.append(predicted_labels)
        S.append(predicted_scores)
    AUC = roc_auc_score(T, S)
    precision = precision_score(T, Y)
    recall = recall_score(T, Y)
    return AUC, precision, recall

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def shuffle_dataset(dataset):
    np.random.shuffle(dataset)
    return dataset

def load_tensor(file_name, dtype, device):
    data = np.load(file_name + '.npy', allow_pickle=True)
    return [dtype(d).to(device) for d in data]

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def main():
    '''Hyperparameters.'''
    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
            lr, lr_decay, decay_interval, weight_decay, iteration, setting) = sys.argv[1:]
    (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval, iteration) = \
            map(int, [dim, layer_gnn, window, layer_cnn, layer_output, decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    '''CPU or GPU.'''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('The code uses %s...' % device)

    '''Load preprocessed data.'''
    dir_input = ('../dataset/%s/input/radius%s_ngram%s/' % (DATASET, radius, ngram))

    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor, device)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor, device)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor, device)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor, device)

    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pkl')
    word_dict = load_pickle(dir_input + 'word_dict.pkl')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    '''Create a dataset and split it into train/dev/test.'''
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    dataset = shuffle_dataset(dataset)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    '''Set a model.'''
    model = CompoundProteinInteractionPrediction(n_fingerprint, n_word, dim, window,
            layer_gnn, layer_cnn, layer_output).to(device)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)

    model.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    '''Output files.'''
    file_AUCs = '../output/result/AUCs--%s.txt' % setting
    file_model = '../output/model/%s' % setting

    AUCs = pd.DataFrame(columns=['Epoch', 'Time(sec)', 'Loss_train', 
        'AUC_dev', 'AUC_test', 'Precision_test', 'Recall_test'])

    '''Start training.'''
    print('Training...')
    start = timeit.default_timer()

    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            model.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = train(model, dataset_train)
        AUC_dev = test(model, dataset_dev)[0]
        AUC_test, precision_test, recall_test = test(model, dataset_test)

        end = timeit.default_timer()
        time = end - start

        values = [epoch, time, loss_train, AUC_dev, AUC_test, precision_test, recall_test]
        AUCs = AUCs.append(dict((k,v) for k,v in zip(AUCs.columns, values)), ignore_index=True)

        AUCs.to_csv(file_AUCs, index=False)
        save_model(model, file_model)

        print(AUCs.tail(10))

if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    main()
