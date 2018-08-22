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
from scipy.sparse import coo_matrix


class GATEdgePhaseOne(nn.Module):
    def __init__(self):
        super(GATEdgePhaseOne, self).__init__()

    def forward(self, u, v, edge):
        attention = F.leaky_relu(u['a2'] + v['a1'])
        return {'attention': attention}

class GATEdgePhaseTwo(nn.Module):
    def __init__(self):
        super(GATEdgePhaseTwo, self).__init__()

    def forward(self, u, v, edge):
        attention = torch.exp(edge['attention'] - v['a_max'])
        return {'attention': attention}

class GATEdgePhaseThree(nn.Module):
    def __init__(self, attn_drop):
        super(GATEdgePhaseThree, self).__init__()
        self.attn_drop = attn_drop

    def forward(self, u, v, edge):
        attention = edge['attention'] / v['a_sum']
        if self.attn_drop != 0.0:
            attention = F.dropout(attention, self.attn_drop)
        return {'attention': attention}


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
        return {'h': h, 'ft': ft, 'a1': a1, 'a2': a2}


class GATFinalize(nn.Module):
    def __init__(self, indim, hiddendim, activation, residual):
        super(GATFinalize, self).__init__()
        self.activation = activation
        self.residual = residual
        self.residual_fc = None
        if residual:
            if indim != hiddendim:
                self.residual_fc = nn.Linear(indim, hiddendim)

    def forward(self, node):
        ret = node['ft']
        h = node['h']
        if self.residual:
            if self.residual_fc is not None:
                ret = self.residual_fc(h) + ret
            else:
                ret = h + ret
        return self.activation(ret)


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
                 residual,
                 use_cuda):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prp = nn.ModuleList()
        self.fnl = nn.ModuleList()
        self.src, self.dst = self.g.cached_graph.edges()
        self.use_cuda = use_cuda
        # edge update
        self.attn_phase1 = GATEdgePhaseOne()
        self.attn_phase2 = GATEdgePhaseTwo()
        self.attn_phase3 = GATEdgePhaseThree(attn_drop)
        # calc normalization factor
        self.attn_sum = lambda node, accum: {'a_sum': accum['a_sum']}
        self.attn_mul = lambda node, accum: {'ft': accum['ft']}
        # input projection (no residual)
        for _ in range(num_heads):
            self.prp.append(GATPrepare(in_dim, num_hidden, in_drop))
            self.fnl.append(GATFinalize(in_dim, num_hidden, activation, False))
        # hidden layers
        for l in range(num_layers - 1):
            for _ in range(num_heads):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.prp.append(GATPrepare(num_hidden * num_heads, num_hidden, in_drop))
                self.fnl.append(GATFinalize(num_hidden * num_heads,
                                            num_hidden, activation, residual))
        # output projection
        self.prp.append(GATPrepare(num_hidden * num_heads, num_classes, in_drop))
        self.fnl.append(GATFinalize(num_hidden * num_heads,
                                    num_classes, activation, residual))
        # sanity check
        assert len(self.prp) == self.num_layers * self.num_heads + 1
        assert len(self.fnl) == self.num_layers * self.num_heads + 1

        if use_cuda:
            self.tensor1 = torch.ones((len(g), 1)).cuda()
        else:
            self.tensor1 = torch.ones((len(g), 1))

    def forward_one_head(self, update_func):
        # calc unnormalized attention value
        self.g.update_edge(edge_func=self.attn_phase1, batchable=True)
        # calc a_max on cpu
        attention = self.g.get_e_repr()['attention']
        a_detached = attention.detach().view(-1)
        if self.use_cuda:
            a_detached = a_detached.cpu()
        coo = coo_matrix((a_detached.numpy(), (self.src, self.dst)), shape=[len(self.g), len(self.g)])
        a_max = coo.max(axis=0)
        a_max = torch.from_numpy(a_max.toarray()).view(-1, 1)
        if self.use_cuda:
            a_max = a_max.cuda()
        self.g.set_n_repr({'a_max': a_max})
        # normalize a by deduct max
        self.g.update_edge(edge_func=self.attn_phase2, batchable=True)
        # pop out useless features
        self.g.pop_n_repr('a1')
        self.g.pop_n_repr('a2')
        self.g.pop_n_repr('a_max')
        # stash features not needed for spmv
        h = self.g.pop_n_repr('h')
        ft = self.g.pop_n_repr('ft')
        # set 1 tensor for spmv
        self.g.set_n_repr({'a_sum': self.tensor1})
        # call spmv
        self.g.update_all('src_mul_edge', 'sum', self.attn_sum, batchable=True)
        # calculate attential and dropout
        self.g.update_edge(edge_func=self.attn_phase3, batchable=True)
        # pop out useless tensors
        self.g.pop_n_repr('a_sum')
        # restore node state
        self.g.set_n_repr({'ft': ft})
        # call spmv
        self.g.update_all('src_mul_edge', 'sum', self.attn_mul, batchable=True)
        self.g.set_n_repr({'h': h})
        return update_func(self.g.get_n_repr())

    def forward(self, features):
        last = features
        for l in range(self.num_layers):
            curr = []
            for hid in range(self.num_heads):
                i = l * self.num_heads + hid
                # prepare
                self.g.set_n_repr(self.prp[i](last))
                head = self.forward_one_head(self.fnl[i])
                curr.append(head)

            # merge all the heads
            last = torch.cat(curr, dim=1)
        # output projection
        self.g.set_n_repr(self.prp[-1](last))
        return self.forward_one_head(self.fnl[-1])


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
                args.residual,
                cuda)

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
