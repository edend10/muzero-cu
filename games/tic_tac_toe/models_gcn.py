import numpy as np
import pandas as pd
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import networkx as nx

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h' : h} # can't change latent dim otherwise this fails on the output shape != 9


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Representation(nn.Module):
    def __init__(self, num_blocks, channels_in, latent_dim=3):
        super(Representation, self).__init__()
        self.gcn1 = GCN(channels_in * 3 * 3, latent_dim * 3 * 3, F.relu)
        self.graph = self.create_graph()
        self.latent_dim = latent_dim
        # self.gcn2 = GCN(16, 7, None)

    def forward(self, features):
        features = features.reshape(9,)
        x = self.gcn1(self.graph, features)
        # x = self.gcn2(g, x)
        return x.reshape(1, self.latent_dim, 3, 3)

    def create_graph(self):
        df = pd.DataFrame(get_adjacency_matrix())
        g = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
        g.remove_edges_from(nx.selfloop_edges(g))
        g = DGLGraph(g)
        g.add_edges(g.nodes(), g.nodes())
        return g


def get_adjacency_matrix():
    return np.array([[0., 1., 2., 1., 1., 2., 2., 2., 2.],
       [1., 0., 1., 1., 1., 1., 2., 2., 2.],
       [2., 1., 0., 2., 1., 1., 2., 2., 2.],
       [1., 1., 2., 0., 1., 2., 1., 1., 2.],
       [1., 1., 1., 1., 0., 1., 1., 1., 1.],
       [2., 1., 1., 2., 1., 0., 2., 1., 1.],
       [2., 2., 2., 1., 1., 2., 0., 1., 2.],
       [2., 2., 2., 1., 1., 1., 1., 0., 1.],
       [2., 2., 2., 2., 1., 1., 2., 1., 0.]])


