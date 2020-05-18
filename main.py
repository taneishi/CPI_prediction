import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import argparse
import timeit

from model import CompoundProteinInteractionModel

def train(model, dataset, optimizer, loss_function, epoch):
    model.train()
    train_loss = 0
    for index, (fingerprints, adjacency, words, interaction) in enumerate(dataset, 1):
        optimizer.zero_grad()
        output = model.forward(fingerprints, adjacency, words)   
        loss = loss_function(output, interaction)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        print('\repoch %3d batch %4d/%4d train_loss %5.3f' % (epoch, index, len(dataset), train_loss / index), end='')

def test(model, dataset, loss_function):
    model.eval()
    test_loss = 0
    y_score, y_true = [], []
    for index, (fingerprints, adjacency, words, interaction) in enumerate(dataset, 1):
        with torch.no_grad():
            output = model.forward(fingerprints, adjacency, words)

        loss = loss_function(output, interaction)
        test_loss += loss.item()
        score = F.softmax(output, 1).cpu()
        y_score.append(score)
        y_true.append(interaction.cpu())

    y_score = np.concatenate(y_score)
    y_pred = [np.argmax(x) for x in y_score]
    y_true = np.concatenate(y_true)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score[:,1])
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(' test_loss %5.3f test_acc %5.3f test_auc %5.3f test_prec %5.3f test_recall %5.3f' % \
            (test_loss / index, acc, auc, prec, recall), end='')

    return test_loss / index

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
    parser.add_argument('--random_seed', default=123)

    args = parser.parse_args()

    print(vars(args))
    
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # CPU or GPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    # Load preprocessed data.
    dataset_train, dataset_test, n_fingerprint, n_word = np.load('dataset/%s.npz' % args.dataset, allow_pickle=True).values()
    
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

    test_losses = [np.inf]

    # Start training
    for epoch in range(args.epochs):
        epoch_start = timeit.default_timer()

        if epoch % args.decay_interval == 0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

        train(model, dataset_train, optimizer, loss_function, epoch)
        test_loss = test(model, dataset_test, loss_function)

        print(' %5.3f sec' % (timeit.default_timer() - epoch_start))

        test_losses.append(test_loss)

        if test_loss < min(test_losses[:-1]):
            torch.save(model.state_dict(), 'model.pth')

        if min(test_losses) < min(test_losses[-10:]):
            break
        
    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    main()
