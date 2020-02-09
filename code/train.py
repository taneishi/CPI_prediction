#!/usr/bin/env python
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
import timeit

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
        loss_total += loss.item()
    return loss_total

def test(model, dataset):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    for fingerprints, adjacency, words, interaction in dataset:
        output = model.forward(fingerprints, adjacency, words)
        
        score = F.softmax(output, 1).cpu().detach().numpy()
        
        predicted_labels = [np.argmax(x) for x in score]
        predicted_scores = [x[1] for x in score]
        
        y_true.append(interaction.cpu().detach().numpy())
        y_pred.append(predicted_labels)
        y_score.append(predicted_scores)

    #auc = roc_auc_score(y_true, y_score)
    #precision = precision_score(y_true, y_pred)
    #recall = recall_score(y_true, y_pred)
    acc = np.equal(y_true, y_pred).sum()
    return acc

def main():
    '''Hyperparameters.'''
    DATASET = 'human'
    # DATASET = 'celegans'

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
    iteration = 30

    setting = '%d-%d-%d-%d-%d-%d-%d-%d-%f-%f-%d-%f' % (
            radius, ngram, dim, layer_gnn, side, window, layer_cnn, layer_output,
            lr, lr_decay, decay_interval, weight_decay)
    
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # CPU or GPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    # Load preprocessed data.
    dataset_train, dataset_test, n_fingerprint, n_word = np.load('dataset.npz', allow_pickle=True).values()
    
    for i in range(len(dataset_train)):
        dataset_train[i,0] = torch.LongTensor(dataset_train[i,0]).to(device)
        dataset_train[i,1] = torch.FloatTensor(dataset_train[i,1]).to(device)
        dataset_train[i,2] = torch.LongTensor(dataset_train[i,2]).to(device)
        dataset_train[i,3] = torch.LongTensor(dataset_train[i,3]).to(device)

    for i in range(len(dataset_test)):
        dataset_test[i,0] = torch.LongTensor(dataset_test[i,0]).to(device)
        dataset_test[i,1] = torch.FloatTensor(dataset_test[i,1]).to(device)
        dataset_test[i,2] = torch.LongTensor(dataset_test[i,2]).to(device)
        dataset_test[i,3] = torch.LongTensor(dataset_test[i,3]).to(device)

    print('train %d test %d' % (len(dataset_train), len(dataset_test)))
    
    # Set a model.
    model = CompoundProteinInteractionPrediction(n_fingerprint, n_word, dim, window, layer_gnn, layer_cnn, layer_output)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Output files.
    file_AUCs = '../output/result/AUCs--%s.txt' % setting
    file_model = '../output/model/%s' % setting

    # Start training.
    print('Training...')
    print('%5s%12s%12s%12s' % ('epoch', 'train_loss', 'test_acc', 'time(sec)'))

    for epoch in range(1, iteration+1):
        epoch_start = timeit.default_timer()

        if epoch % decay_interval == 0:
            model.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = train(model, dataset_train)
        test_acc = test(model, dataset_test)

        time = timeit.default_timer() - epoch_start

        print('%5d%12.4f%12.4f%12.4f' % (epoch, loss_train, test_acc, time))
        
    torch.save(model.state_dict(), file_model)

if __name__ == '__main__':
    main()
