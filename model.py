import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
import gensim
import torch_scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# 忽略与Levenshtein子模块相关的UserWarning
warnings.filterwarnings("ignore", message="The gensim.similarities.levenshtein submodule is disabled")


class CustomGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True):
        super(CustomGCNConv, self).__init__(aggr='add', flow='target_to_source')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, edge_index, edge_weight, self_loop_weights=None):
        # 如果你的图没有自环，你可能想要添加它们
        if self_loop_weights is not None:
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(0),
                                                     fill_value=self_loop_weights)

            # 先进行线性变换
        x = self.lin(x)

        # 然后进行消息传递
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # x_j是邻居节点的特征
        # edge_weight是对应的边权重
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        # 在这里，我们不需要额外的操作，因为所有的转换已经在message函数中完成
        return aggr_out


def random_walk(G, v, l):
    """
    随机游走。
    G: 图
    v: 游走的根节点
    l: 游走的序列的长度
    """
    walk_seq = [v]
    while len(walk_seq) < l:
        node = walk_seq[-1]
        neighbors = list(G.neighbors(node))
        selected_neighbors = list(set(neighbors) - set(walk_seq))
        if len(selected_neighbors) > 0:
            # 当前节点有邻居，且存在不在walk_seq中的邻居，随机选择一个邻居加入
            walk_seq.append(random.choice(selected_neighbors))
        else:
            break
    return walk_seq


def deep_walk(G, t, l, d, w, epochs=5):
    """
    深度随机游走。
    G: 图
    t: 每个节点随机游走的次数
    l: 每次随机游走的序列长度
    d: 节点向量的维度
    w: skipgram的窗口大小
    epochs: skipgram: 训练论述
    """
    walk_seq_list = []
    nodes = list(G.nodes)
    for i in range(t):
        # 打乱
        # random.shuffle(nodes)
        for node in nodes:
            walk_seq = random_walk(G, node, l)
            walk_seq_list.append(walk_seq)


    model = gensim.models.Word2Vec(sentences=walk_seq_list, vector_size=d, window=w, workers=5,
                                   max_vocab_size=len(G), min_count=0, epochs=epochs)
    return model


class LstmEmbedding(nn.Module):
    def __init__(self, vocab_size, num_features):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, num_features)
        self.fc1 = nn.Linear(num_features, num_features)
        self.lstm = nn.LSTM(num_features, num_features, num_layers=2)

    def forward(self, X):
        X = F.relu(self.embedding(X).permute(1, 0, 2)) # u,l,d
        X = self.fc1(X)
        h, _ = self.lstm(X)[1] # h: 2, u, d
        u = h.permute(1, 0, 2).sum(dim=1)  # u, d

        return u
    

# class QNet(nn.Module):
#     def __init__(self, vocab_size, num_features):
#         super().__init__()
#         self.lm = LstmEmbedding(vocab_size, num_features) # u, d
#         # self.embedding = nn.Embedding(num_nodes, num_features) # u, d
#         # self.a1 = nn.Linear(1, 1, bias=False)
#         # self.a2 = nn.Linear(1, 1, bias=False)
#         # self.a3 = nn.Linear(1, 1, bias=False)
#         self.a5 = nn.Linear(num_features, num_features, bias=False)
#         self.a6 = nn.Linear(num_features, num_features, bias=False)
#         self.a7 = nn.Linear(2 * num_features, num_features)
#         self.a8 = nn.Linear(num_features, num_features, bias=False)
#         self.a9 = nn.Linear(num_features, num_features, bias=False)
#
#         self.fc1 = nn.Linear(num_features, num_features)
#         self.convs1 = nn.ModuleList()
#         self.convs2 = nn.ModuleList()
#         self.alphas1 = nn.ModuleList()
#         self.alphas2 = nn.ModuleList()
#         self.alphas3 = nn.ModuleList()
#         self.alphas4 = nn.ModuleList()
#         for _ in range(3):
#             self.convs1.append(CustomGCNConv(num_features, num_features, bias=False))
#             self.convs2.append(CustomGCNConv(num_features, num_features, bias=False))
#             self.alphas1.append(nn.Linear(1, 1, bias=False))
#             self.alphas2.append(nn.Linear(1, 1, bias=False))
#             self.alphas3.append(nn.Linear(num_features, num_features, bias=False))
#             self.alphas4.append(nn.Linear(num_features, num_features, bias=False))
#         # self.conv1 = CustomGCNConv(num_features, num_features, bias=False)
#         # self.conv2 = CustomGCNConv(num_features, num_features, bias=False)
#         # self.conv3 = CustomGCNConv(num_features, num_features, bias=False)
#         self.b1 = nn.Linear(num_features, num_features, bias=False)
#         self.b2 = nn.Linear(num_features, num_features, bias=False)
#         self.b3 = nn.Linear(num_features, num_features, bias=False)
#
#         self.fc2 = nn.Linear(num_features, num_features, bias=False)
#         self.fc3 = nn.Linear(2 * num_features, 1)
#         # self.bn1 = nn.BatchNorm1d(num_features)
#
#         self.a4 = nn.Linear(num_features, num_features, bias=False)
#
#     def forward(self, X1, X2, edge_index, edge_weights, batch, states):
#         # states: b, n
#         # X1: u, d
#         # X2: u(0,1,2)
#         # print(X1.shape)
#         # Y1 = self.lm(X1)
#         # Y2 = self.embedding(X2) # u, d
#         # print(Y1.shape, Y2.shape)
#         # Y = torch.cat([self.a5(Y1), self.a6(X2)], dim=1)
#         Y = X2
#         # Y = F.relu(self.fc1(Y))
#         # Y可以使用GNN=>Y, edge_index
#         # print(edge_index)
#         # num_graphs = len(states)
#         states = torch.tensor([s for state in states for s in state], dtype=torch.float, device=X1.device).view(-1, 1)
#         Y1 = Y[:, : Y.shape[1] // 2]
#         Y2 = Y[:, Y.shape[1] // 2:]
#         for conv1, conv2, alpha1, alpha2, alpha3, alpha4 in zip(self.convs1, self.convs2, self.alphas1, self.alphas2,
#                                                                 self.alphas3, self.alphas4):
#             Y1 = F.leaky_relu(alpha3(F.leaky_relu(conv1(Y1, edge_index[[1, 0]], edge_weights, 1) + alpha1(states), 0.2)), 0.2)
#             Y2 = F.leaky_relu(alpha4(F.leaky_relu(conv2(Y2, edge_index, edge_weights, 1) + alpha2(states), 0.2)), 0.2)
#
#         W = F.leaky_relu(self.a7(torch.cat([self.a5(Y1), self.a6(Y2)], dim=-1)), 0.2)
#
#         activate_X = torch_scatter.scatter_add(W[states.view(-1) == 1], batch[states.view(-1) == 1], dim=0, dim_size=batch[-1].item() + 1).repeat_interleave(
#             torch_scatter.scatter_add(torch.ones(batch.size(0), dtype=torch.long, device=batch.device), batch), dim=0)
#         X_activte = self.fc2(activate_X)
#         q = self.fc3(torch.cat([self.a4(W), X_activte], dim=1)).squeeze() #b'
#         return q

class QNet(nn.Module):
    def __init__(self, num_features, T=3, aggr_direction='all'):
        super(QNet, self).__init__()
        self.T = T
        self.aggr_direction = aggr_direction
        # self.bn1 = nn.BatchNorm1d(2 * num_features)

        self.layers = nn.ModuleList()
        for i in range(T):
            self.layers.append(nn.ModuleDict())
            if self.aggr_direction == 'in':
                self.layers[i][f'conv{i}_0'] = CustomGCNConv(2 * num_features, 2 * num_features, bias=False)
                self.layers[i][f'alpha{i}_0'] = nn.Linear(1, 1, bias=False)
                self.layers[i][f'alpha{i}_1'] = nn.Linear(2 * num_features, 2 * num_features)
            elif self.aggr_direction == 'out':
                self.layers[i][f'conv{i}_1'] = CustomGCNConv(2 * num_features, 2 * num_features, bias=False)
                self.layers[i][f'alpha{i}_2'] = nn.Linear(1, 1, bias=False)
                self.layers[i][f'alpha{i}_3'] = nn.Linear(2 * num_features, 2 * num_features)
            elif self.aggr_direction == 'all':
                self.layers[i][f'conv{i}_0'] = CustomGCNConv(num_features, num_features, bias=False)
                self.layers[i][f'alpha{i}_0'] = nn.Linear(1, 1, bias=False)
                self.layers[i][f'alpha{i}_1'] = nn.Linear(num_features, num_features)
                self.layers[i][f'conv{i}_1'] = CustomGCNConv(num_features, num_features, bias=False)
                self.layers[i][f'alpha{i}_2'] = nn.Linear(1, 1, bias=False)
                self.layers[i][f'alpha{i}_3'] = nn.Linear(num_features, num_features)
            else:
                raise ValueError('不支持的aggr_direction的值。')
            # self.bn1s.append(nn.BatchNorm1d(num_features))
            # self.bn2s.append(nn.BatchNorm1d(num_features))

        if self.aggr_direction == 'all':
            self.theta0 = nn.Linear(num_features, num_features, bias=False)
            self.theta1 = nn.Linear(num_features, num_features, bias=False)
        self.theta2 = nn.Linear(2 * num_features, num_features)

        self.delta0 = nn.Linear(num_features, num_features, bias=False)
        self.delta1 = nn.Linear(num_features, num_features, bias=False)
        self.delta2 = nn.Linear(2 * num_features, num_features, bias=False)
        self.delta3 = nn.Linear(num_features, 1)

    def forward(self, x, edge_index, edge_weights, batch, states):
        states = torch.tensor([s for state in states for s in state], dtype=torch.float, device=x.device)

        selected_nodes = states == 1
        # x2: 建模节点传播能力
        # x1: 建模节点的状态情况
        # x = self.bn1(x)
        if self.aggr_direction == 'all':
            x1 = x[:, : x.shape[1] // 2]
            x2 = x[:, x.shape[1] // 2:]
        # t轮聚合节点激活情况信息得到节点的激活状态情况
        for i in range(self.T):
            if self.aggr_direction == 'in':
                # 建模节点激活状态情况概率states-1)
                aggr_in_neighbor_x = self.layers[i][f'conv{i}_0'](x, edge_index[[1, 0]], edge_weights, 0.0 if i == 0 else 1.0)
                x1_state = F.leaky_relu(aggr_in_neighbor_x + self.layers[i][f'alpha{i}_0'](states.view(-1, 1)), 0.2).float()
                x = F.leaky_relu(self.layers[i][f'alpha{i}_1'](x1_state), 0.2)
            elif self.aggr_direction == 'out':
                # 建模节点传播能力
                aggr_out_neighbor_x = self.layers[i][f'conv{i}_1'](x, edge_index, edge_weights, 0.0 if i == 0 else 1.0)
                x2_state = F.leaky_relu(aggr_out_neighbor_x + self.layers[i][f'alpha{i}_2'](states.view(-1, 1)), 0.2).float()
                x = F.leaky_relu(self.layers[i][f'alpha{i}_3'](x2_state), 0.2)
            else:
                aggr_in_neighbor_x = self.layers[i][f'conv{i}_0'](x1, edge_index[[1, 0]], edge_weights, 0.0 if i == 0 else 1.0)
                x1_state = F.leaky_relu(aggr_in_neighbor_x + self.layers[i][f'alpha{i}_0'](states.view(-1, 1)), 0.2).float()
                x1 = F.leaky_relu(self.layers[i][f'alpha{i}_1'](x1_state), 0.2)
                aggr_out_neighbor_x = self.layers[i][f'conv{i}_1'](x2, edge_index, edge_weights, 0.0 if i == 0 else 1.0)
                x2_state = F.leaky_relu(aggr_out_neighbor_x + self.layers[i][f'alpha{i}_2'](states.view(-1, 1)), 0.2).float()
                x2 = F.leaky_relu(self.layers[i][f'alpha{i}_3'](x2_state), 0.2)

        if self.aggr_direction == 'all':
            # 拼接得到最终向量
            x = F.leaky_relu(self.theta2(torch.cat([self.theta0(x1), self.theta1(x2)], dim=-1)), 0.2)
        else:
            x = F.leaky_relu(self.theta2(x), 0.2)

        # 计算q
        x_s = torch_scatter.scatter_add(x[selected_nodes], batch[selected_nodes], dim=0,
                                        dim_size=batch[-1].item() + 1).repeat_interleave(
            torch_scatter.scatter_add(torch.ones(batch.shape[0], dtype=torch.long, device=batch.device), batch), dim=0)
        # x_sum = torch_scatter.scatter_add(x, batch, dim=0, dim_size=batch[-1].item() + 1).repeat_interleave(
        #     torch_scatter.scatter_add(torch.ones(batch.shape[0], dtype=torch.long, device=batch.device), batch), dim=0)
        x = F.leaky_relu(torch.cat([self.delta0(x), self.delta1(x_s)], dim=-1), 0.2)
        x = F.leaky_relu(self.delta2(x), 0.2)
        q = self.delta3(x).view(-1)
        return q



