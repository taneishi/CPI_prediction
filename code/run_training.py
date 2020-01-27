#!/usr/bin/env python
import numpy as np
import pickle
import timeit
import sys
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self, n_fingerprint, n_word, dim, window, layer_gnn, layer_cnn, layer_output):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim, layer_gnn)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)] * layer_gnn)
        self.W_cnn = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2*window+1, stride=1, padding=window)] * layer_cnn)
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)] * layer_output)
        self.W_interaction = nn.Linear(2*dim, 2)

    def gnn(self, xs, A):
        for i in range(len(self.W_gnn)):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs):
        '''The attention mechanism is applied to the last layer of CNN.'''
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(len(self.W_cnn)):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, fingerprints, adjacency, words):
        # Compound vector with GNN.
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency)

        # Protein vector with attention-CNN.
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector, word_vectors)

        # Concatenate the above two vectors and output the interaction.
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(len(self.W_out)):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction

def train(model, dataset):
    model.train()
    loss_total = 0
    for fingerprints, adjacency, words, y_true in dataset:
        y_pred = model.forward(fingerprints, adjacency, words)   
        loss = F.cross_entropy(y_pred, y_true)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        loss_total += loss.cpu().data.numpy()
    return loss_total

def test(model, dataset):
    model.eval()
    T, Y, S = [], [], []
    for fingerprints, adjacency, words, y_true in dataset:
        y_pred = model.forward(fingerprints, adjacency, words)
        
        correct_labels = y_true.cpu().data.numpy()
        ys = F.softmax(y_pred, 1).cpu().data.numpy()
        
        predicted_labels = [np.argmax(x) for x in ys]
        predicted_scores = [x[1] for x in ys]
        
        T.append(correct_labels)
        Y.append(predicted_labels)
        S.append(predicted_scores)
    AUC = roc_auc_score(T, S)
    precision = precision_score(T, Y)
    recall = recall_score(T, Y)
    return AUC, precision, recall

def load_tensor(file_name, dtype, device):
    data = np.load(file_name + '.npy', allow_pickle=True)
    return [dtype(d).to(device) for d in data]

def main():
    '''Hyperparameters.'''
    DATASET = 'human'
    # DATASET = 'celegans'
    # DATASET = 'yourdata'

    # radius = 1
    radius = 2
    # radius = 3

    # ngram = 2
    ngram = 3

    dim = 10
    layer_gnn = 3
    side = 5
    window = 2*side+1
    layer_cnn = 3
    layer_output = 3
    
    lr = 1e-3
    lr_decay = 0.5
    decay_interval = 10
    weight_decay = 1e-6
    iteration = 100
    
    setting = '%d-%d-%d-%d-%d-%d-%d-%d-%f-%f-%d-%f' % (
            radius, ngram, dim, layer_gnn, slide, window, layer_cnn, layer_output,
            lr, lr_decay, decay_interval, weight_decay)
    
    # CPU or GPU.
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('The code uses %s...' % device)

    # Load preprocessed data.
    dir_input = ('../dataset/%s/input/radius%s_ngram%s/' % (DATASET, radius, ngram))

    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor, device)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor, device)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor, device)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor, device)

    fingerprint_dict = pickle.load(open(dir_input + 'fingerprint_dict.pkl', 'rb'))
    word_dict = pickle.load(open(dir_input + 'word_dict.pkl', 'rb'))
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    # Create a dataset and split it into train/dev/test.
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    np.random.shuffle(dataset)
    dataset_train, dataset_test = train_test_split(dataset, train_size=0.8, test_size=0.2, stratify=interactions)
    print(len(dataset), len(dataset_train), len(dataset_test))
    
    # Set a model.
    model = CompoundProteinInteractionPrediction(n_fingerprint, n_word, dim, window, layer_gnn, layer_cnn, layer_output)
    model = model.to(device)

    #if torch.cuda.is_available():
    #    model = torch.nn.DataParallel(model)

    model.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Output files.
    file_AUCs = '../output/result/AUCs--%s.txt' % setting
    file_model = '../output/model/%s' % setting

    columns = ['Epoch', 'Time(sec)', 'Loss_train', 'AUC_test', 'Prec_test', 'Recall_test']

    # Start training.
    print('Training...')
    print(''.join(map(lambda x: '%12s' % x, columns)))
    
    start = timeit.default_timer()

    for epoch in range(1, iteration+1):
        if epoch % decay_interval == 0:
            model.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = train(model, dataset_train)
        AUC_test, precision_test, recall_test = test(model, dataset_test)

        time = timeit.default_timer() - start
        start = start + time

        values = [epoch, time, loss_train, AUC_test, precision_test, recall_test]
        print('%12s' % epoch + ''.join(map(lambda x: '%12.4f' % x, values[1:])))
        
    torch.save(model.state_dict(), file_model)

if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    main()
