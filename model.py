import torch.nn as nn
import torch.nn.functional as F
import torch

class CompoundProteinInteractionModel(nn.Module):
    def __init__(self, n_fingerprint, n_word, args):
        super(CompoundProteinInteractionModel, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, args.dim, args.layer_gnn)
        self.embed_word = nn.Embedding(n_word, args.dim)
        self.W_gnn = nn.ModuleList([nn.Linear(args.dim, args.dim)] * args.layer_gnn)
        cnn = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2*args.window+1, stride=1, padding=args.window)
        self.W_cnn = nn.ModuleList([cnn] * args.layer_cnn)
        self.W_attention = nn.Linear(args.dim, args.dim)
        self.W_out = nn.ModuleList([nn.Linear(2*args.dim, 2*args.dim)] * args.layer_output)
        self.W_interaction = nn.Linear(2*args.dim, 2)

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
