import numpy as np
import pandas as pd
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import networkx as nx

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class Conv2DBlock(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, bn=False, relu=False):
        super().__init__()
        self.conv = nn.Conv2d(filters_in, filters_out, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn = None
        self.relu=None
        if bn:
            self.bn = nn.BatchNorm2d(filters_out)
        if relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        h = self.conv(x)
        if self.bn:
            h = self.bn(h)
        if self.relu:
            h = self.relu(h)
        return h


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
        self.graph = create_graph()
        self.latent_dim = latent_dim
        # self.gcn2 = GCN(16, 7, None)

    def forward(self, features):
        features = features.reshape(9,)
        x = self.gcn1(self.graph, features)
        # x = self.gcn2(g, x)
        return x.reshape(1, self.latent_dim, 3, 3)


class Prediction(nn.Module):
    def __init__(self, num_blocks, channels_in, size_x, size_y, policy_output_size, latent_dim=4):
        super().__init__()
        latent_size = latent_dim * size_x * size_y
        self.gcn1 = GCN(channels_in * size_x * size_y, latent_size, F.relu)
        self.fc_output_policy_block = nn.Sequential(
            nn.Linear(latent_size, policy_output_size),
            nn.Softmax(dim=-1)
        )
        self.fc_output_value_block = nn.Sequential(
            nn.Linear(latent_size, 1),
            nn.Tanh()
        )

        self.graph = create_graph()
        self.latent_dim = latent_dim
        # self.conv0 = Conv2DBlock(channels_in, latent_dim, kernel_size=3, bn=True, relu=True)
        # self.residual_blocks = nn.ModuleList([ResidualBlock(latent_dim)] * num_blocks)

    def forward(self, features):
        features = features.reshape(9, )
        x = self.gcn1(self.graph, features)

        policy = self.fc_output_policy_block(x)
        value = self.fc_output_value_block(x)

        return policy, value


class Dynamics(nn.Module):
    def __init__(self, num_blocks, size_x, size_y, state_channels_in, action_channels_in, latent_dim=4):
        super().__init__()
        latent_size = latent_dim * size_x * size_y
        self.conv0 = Conv2DBlock(state_channels_in + action_channels_in, latent_dim, 3, bn=True, relu=True)
        self.gcn1 = GCN(latent_size, latent_size, F.relu)
        self.graph = create_graph()
        self.latent_dim = latent_dim
        # self.gcn2 = GCN(16, 7, None)

        self.fc_output_reward_block = nn.Sequential(
            nn.Linear(latent_size, 1),
            nn.Tanh()
        )

    def forward(self, x_tuple):
        x = torch.cat(x_tuple, dim=1)
        features = self.conv0(x)

        features = features.reshape(9, )
        x = self.gcn1(self.graph, features)

        state_output = x.reshape(1, self.latent_dim, 3, 3)

        reward_output = self.fc_output_reward_block(x)

        return state_output, reward_output


def create_graph():
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


