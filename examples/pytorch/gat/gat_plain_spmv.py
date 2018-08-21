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
import networkx as nx
from dgl.data import register_data_args, load_data
from dgl.backend import spmm_grad


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
        return h, ft, a1, a2


class GATFinalize(nn.Module):
    def __init__(self, indim, hiddendim, activation, residual):
        super(GATFinalize, self).__init__()
        self.activation = activation
        self.residual = residual
        self.residual_fc = None
        if residual:
            if indim != hiddendim:
                self.residual_fc = nn.Linear(indim, hiddendim)

    def forward(self, h, ft):
        ret = ft
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
        self.num_node = len(g)
        self.coo = nx.to_scipy_sparse_matrix(g, nodelist=range(self.num_node), format='coo')
        self.num_edge = len(self.coo.data)
        self.src = torch.LongTensor(self.coo.row)
        self.dst = torch.LongTensor(self.coo.col)
        self.indices = torch.stack([self.src, self.dst])
        self.one_tensor = torch.ones((self.num_node, 1))
        self.attn_drop = attn_drop
        self.use_cuda = use_cuda
        if use_cuda:
            self.src = self.src.cuda()
            self.dst = self.dst.cuda()
            self.indices = self.indices.cuda()
            self.one_tensor = self.one_tensor.cuda()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prp = nn.ModuleList()
        self.fnl = nn.ModuleList()
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


    def compute_one_head(self, last, prepare, finalize):
        # calc unnormalized attention value
        h, ft, a1, a2 = prepare(last)
        a_v = torch.index_select(a1, 0, self.dst)
        a_u = torch.index_select(a2, 0, self.src)
        a = F.leaky_relu(a_v + a_u)
        a_detached = a.detach().view(-1)
        if self.use_cuda:
            a_detached = a_detached.cpu()
        self.coo.data = a_detached.numpy()
        a_max = self.coo.max(axis=0).toarray()
        a_max = torch.from_numpy(a_max).view(-1, 1)
        if self.use_cuda:
            a_max = a_max.cuda()
        a_max = torch.index_select(a_max, 0, self.dst)
        unnormalized_a = torch.exp(a-a_max)
        a_sum = spmm_grad(self.indices, unnormalized_a.view(-1), self.one_tensor, [self.num_node, self.num_node])
        a_sum = torch.index_select(a_sum, 0, self.dst)
        attention = unnormalized_a / a_sum
        if self.attn_drop != 0.0:
            attention = F.dropout(attention, self.attn_drop)
        ft = spmm_grad(self.indices, attention.view(-1), ft, [self.num_node, self.num_node])
        return finalize(h, ft)

    def forward(self, features):
        last = features
        for l in range(self.num_layers):
            curr = []
            for hid in range(self.num_heads):
                i = l * self.num_heads + hid
                head = self.compute_one_head(last, self.prp[i], self.fnl[i])
                curr.append(head)
            # merge all the heads
            last = torch.cat(curr, dim=1)

        # output projection
        return self.compute_one_head(last, self.prp[-1], self.fnl[-1])


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
    g = data.graph

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
