"""
Graph Attention Networks
Paper: https://arxiv.org/abs/1710.10903
Code: https://github.com/PetarV-/GAT

GAT with batch processing
"""

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

def gat_message(src, edge):
    return {'ft' : src['ft'], 'a2' : src['a2']}

class GATReduce(nn.Module):
    def __init__(self, attn_drop):
        super(GATReduce, self).__init__()
        self.attn_drop = attn_drop

    def forward(self, node, msgs):
        a1 = torch.unsqueeze(node['a1'], 1)  # shape (B, 1, 1)
        a2 = msgs['a2'] # shape (B, deg, 1)
        ft = msgs['ft'] # shape (B, deg, D)
        # attention
        a = a1 + a2  # shape (B, deg, 1)
        e = F.softmax(F.leaky_relu(a), dim=1)
        if self.attn_drop != 0.0:
            e = F.dropout(e, self.attn_drop)
        return torch.sum(e * ft, dim=1) # shape (B, D)

class GATFinalize(nn.Module):
    def __init__(self, headid, indim, hiddendim, activation, residual):
        super(GATFinalize, self).__init__()
        self.headid = headid
        self.activation = activation
        self.residual = residual
        self.residual_fc = None
        if residual:
            if indim != hiddendim:
                self.residual_fc = nn.Linear(indim, hiddendim)

    def forward(self, node, accum):
        ret = accum
        if self.residual:
            if self.residual_fc is not None:
                ret = self.residual_fc(node['h']) + ret
            else:
                ret = node['h'] + ret
        return {'head%d' % self.headid : self.activation(ret)}

class GATPrepare(nn.Module):
    def __init__(self, indim, hiddendim, drop):
        super(GATPrepare, self).__init__()
        self.fc = nn.Linear(indim, hiddendim)
        self.drop = drop
        self.attn_l = nn.Linear(hiddendim, 1)
        self.attn_r = nn.Linear(hiddendim, 1)

    def forward(self, feats):
        h = feats
        if self.drop != 0.0:
            h = F.dropout(h, self.drop)
        ft = self.fc(h)
        a1 = self.attn_l(ft)
        a2 = self.attn_r(ft)
        return {'h' : h, 'ft' : ft, 'a1' : a1, 'a2' : a2}

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_heads,
                 activation,
                 in_drop,
                 attn_drop,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prp = nn.ModuleList()
        self.red = nn.ModuleList()
        self.fnl = nn.ModuleList()
        # input projection (no residual)
        for hid in range(num_heads):
            self.prp.append(GATPrepare(in_dim, num_hidden, in_drop))
            self.red.append(GATReduce(attn_drop))
            self.fnl.append(GATFinalize(hid, in_dim, num_hidden, activation, False))
        # hidden layers
        for l in range(num_layers - 1):
            for hid in range(num_heads):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.prp.append(GATPrepare(num_hidden * num_heads, num_hidden, in_drop))
                self.red.append(GATReduce(attn_drop))
                self.fnl.append(GATFinalize(hid, num_hidden * num_heads,
                                            num_hidden, activation, residual))
        # output projection
        self.prp.append(GATPrepare(num_hidden * num_heads, num_classes, in_drop))
        self.red.append(GATReduce(attn_drop))
        self.fnl.append(GATFinalize(0, num_hidden * num_heads,
                                    num_classes, activation, residual))
        # sanity check
        assert len(self.prp) == self.num_layers * self.num_heads + 1
        assert len(self.red) == self.num_layers * self.num_heads + 1
        assert len(self.fnl) == self.num_layers * self.num_heads + 1

    def forward(self, features):
        last = features
        for l in range(self.num_layers):
            for hid in range(self.num_heads):
                i = l * self.num_heads + hid
                # prepare
                self.g.set_n_repr(self.prp[i](last))
                # message passing
                self.g.update_all(gat_message, self.red[i], self.fnl[i], batchable=True)
            # merge all the heads
            last = torch.cat(
                    [self.g.pop_n_repr('head%d' % hid) for hid in range(self.num_heads)],
                    dim=1)
        # output projection
        self.g.set_n_repr(self.prp[-1](last))
        self.g.update_all(gat_message, self.red[-1], self.fnl[-1], batchable=True)
        return self.g.pop_n_repr('head0')

def main(args):
    # load and preprocess dataset
    data = load_data(args)

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        mask = mask.cuda()

    # create GCN model
    g = DGLGraph(data.graph)
    if cuda:
        g.set_device(dgl.gpu(args.gpu))

    # create model
    model = GAT(g,
                args.num_layers,
                in_feats,
                args.num_hidden,
                n_classes,
                args.num_heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.residual)

    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # initialize graph
    for_time = []
    back_time = []
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp, labels)
        if epoch >= 3:
            t1 = time.time()

        loss.backward()
        optimizer.step()

        if epoch >= 3:
            t2 = time.time()
            for_time.append(t1 - t0)
            back_time.append(t2 - t1)

        print("Epoch {:05d} | Loss {:.4f} | Forward Time(s) {:.4f} | Backward Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
            epoch, loss.item(), np.mean(for_time), np.mean(back_time),  n_edges / (np.mean(for_time) + np.mean(back_time)) / 1000))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
            help="Which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
            help="number of attentional heads to use")
    parser.add_argument("--num-layers", type=int, default=1,
            help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
            help="size of hidden units")
    parser.add_argument("--residual", action="store_false",
            help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
            help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
            help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
            help="learning rate")
    args = parser.parse_args()
    print(args)

    main(args)
