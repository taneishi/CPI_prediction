import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
import argparse
import timeit

from model import CompoundProteinInteractionModel

def train(model, dataset, optimizer, loss_function, epoch):
    model.train()
    train_loss = 0
    for index, (fingerprints, adjacency, words, y_true) in enumerate(dataset, 1):
        optimizer.zero_grad()
        y_pred = model.forward(fingerprints, adjacency, words)   
        loss = loss_function(y_pred, y_true)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        print('\repoch %3d %4d/%4d train_loss %5.3f' % (epoch, index, len(dataset), train_loss / index), end='')

def test(model, dataset):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    for fingerprints, adjacency, words, interaction in dataset:
        with torch.no_grad():
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
    acc = np.equal(y_true, y_pred).sum() / len(dataset)

    print(' test_acc %5.3f' % acc, end='')

def main():
    '''Hyperparameters.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='human', choices=['human', 'celegans'])
    parser.add_argument('--radius', default=2, choices=[1, 2, 3])
    parser.add_argument('--ngram', default=3, choices=[2, 3])
    parser.add_argument('--dim', default=10)
    parser.add_argument('--layer_gnn', default=3)
    parser.add_argument('--side', default=5)
    parser.add_argument('--window', default=2*5+1) # 2*side+1
    parser.add_argument('--layer_cnn', default=3)
    parser.add_argument('--layer_output', default=3)
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--lr_decay', default=0.5)
    parser.add_argument('--decay_interval', default=10)
    parser.add_argument('--weight_decay', default=1e-6)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--save_path', default='model_pth')

    args = parser.parse_args()

    print(args)
    
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # CPU or GPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    # Load preprocessed data.
    dataset_train, dataset_test, n_fingerprint, n_word = np.load('%s.npz' % args.dataset, allow_pickle=True).values()
    
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
    
    # Set a model
    model = CompoundProteinInteractionModel(n_fingerprint, n_word, args)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_function = F.cross_entropy

    # Start training
    print('Training...')

    for epoch in range(1, args.epochs+1):
        epoch_start = timeit.default_timer()

        if epoch % args.decay_interval == 0:
            optimizer.param_groups[0]['lr'] *= lr_decay

        train(model, dataset_train, optimizer, loss_function, epoch)
        test_acc = test(model, dataset_test)

        print(' %5.3f sec' % (timeit.default_timer() - epoch_start))
        
    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    main()
