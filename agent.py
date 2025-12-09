import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import torch_scatter
from torch_geometric.data import Data, Batch

import rl_utils
from model import QNet, deep_walk
class Agent():
    def __init__(self, num_features, gamma, epsilon, lr, device, target_update=10, n_steps=2, t=10, l=20, w=4, DDQN=True, 
                    aggr_direction='all', graph_name=None, use_deepwalk=True):
        # q网络
        self.q_net = QNet(num_features, aggr_direction=aggr_direction).to(device)
        self.q_net.apply(self.init_weights)
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        # 目标网络
        self.target_q_net = QNet(num_features, aggr_direction=aggr_direction).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.count = 0
        self.target_update = target_update
        self.n_steps = n_steps
        self.graph_init_embedding_dict = {}
        self.t = t
        self.l = l
        self.d = num_features
        self.w = w
        self.graph_name = graph_name
        self.use_deepwalk = use_deepwalk
        self.DDQN = DDQN

    def take_action(self, state, env):
        # 可选节点
        selectable_nodes = list(set(env.graph.nodes()) - set(env.seeds))
        if random.random() < self.epsilon:
            # 随机选一个
            node = random.choice(selectable_nodes)
        else:
            selectable_nodes_t = torch.tensor(selectable_nodes, dtype=torch.long, device=self.device)
            states = [state]
            data = self.get_q_net_input([env.graph])
            self.q_net.eval()
            # with torch.no_grad():
            q_values = self.q_net(data.x, data.edge_index, data.edge_weights, data.batch, states)
            selectable_q_values_sort = q_values[selectable_nodes_t].sort(descending=True).values
            for mq in selectable_q_values_sort:
                max_position = set((q_values == mq).nonzero().view(-1).tolist())
                nodes = list(set(selectable_nodes).intersection(max_position))
                if len(nodes) > 0:
                    #print(nodes)
                    node = random.choice(nodes)
                    break
                else:
                    print(mq,(q_values==mq).nonzero().view(-1))
        return node

    def update(self, states, actions, rewards, next_states, graphs, dones):
        # states = torch.tensor([s for state in states for s in state], dtype=torch.float, device=self.device)
        # next_states = torch.tensor([s for state in next_states for s in state], dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)

        data = self.get_q_net_input(graphs)

        self.q_net.train()
        q_values = self.q_net(data.x, data.edge_index, data.edge_weights, data.batch, states)
        bidx = torch.unique(data.batch)
        len_b = 0
        for b in bidx:
            actions[b] += len_b
            len_b += (data.batch == b).sum()
        q_values = q_values.gather(dim=0, index=actions)
        self.target_q_net.eval()
        if self.DDQN:
            max_q_actions = torch_scatter.scatter_max(self.q_net(data.x, data.edge_index, data.edge_weights,
                                                                       data.batch, states) +
                                                     torch.tensor([ss for s in next_states for ss in s], dtype=torch.float, device=self.device) * -1e6,
                                                     data.batch)[1].clamp_(min=0)
            max_q_values = self.target_q_net(data.x, data.edge_index, data.edge_weights,
                                                                       data.batch, next_states).gather(0, max_q_actions)
        else:
            max_q_values = torch_scatter.scatter_max(self.target_q_net(data.x, data.edge_index, data.edge_weights,
                                                                           data.batch, next_states) +
                                                         torch.tensor([ss for s in next_states for ss in s], dtype=torch.float, device=self.device) * -1e6,
                                                         data.batch)[0].clamp_(min=0)

        q_targets = rewards + self.gamma ** self.n_steps * max_q_values * (1 - dones)
        self.optim.zero_grad()
        loss = F.mse_loss(q_values, q_targets.detach())
        loss.backward()
        self.optim.step()
        self.count += 1
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        return loss.item()

    def get_q_net_input(self, graphs):
        data_list = []
        for i, graph in enumerate(graphs):
            # 简化缓存逻辑
            graph_id = id(graph)
            if graph_id not in self.graph_init_embedding_dict:
                # 直接调用获取嵌入的方法，该方法内部会处理文件缓存
                self.graph_init_embedding_dict[graph_id] = self.get_graph_init_embedding(graph)
            
            init_X_g = self.graph_init_embedding_dict[graph_id]
            graph_edge_index, graph_edge_weights = rl_utils.get_edge_index(graph)
            data = Data(torch.tensor(init_X_g, dtype=torch.float), graph_edge_index, edge_weights=graph_edge_weights)
            data_list.append(data)
    
        batched_data = Batch.from_data_list(data_list)
        return batched_data.to(self.device)

    def get_graph_init_embedding(self, graph):
        if self.use_deepwalk:
            # 先检查文件是否存在，如果存在就直接加载
            if self.graph_name is not None and os.path.exists(f'{self.graph_name}/{self.graph_name}_init_X.npy'):
                init_X = np.load(f'{self.graph_name}/{self.graph_name}_init_X.npy')
                print('读取deepwalk嵌入')
            else:
                # 文件不存在时才训练DeepWalk
                print('训练deepwalk嵌入')
                model = deep_walk(graph, self.t, self.l, self.d * 2, self.w)
                node_vec_index = np.array(sorted(model.wv.key_to_index.items(), key=lambda x: x[0]))[:, 1]
                init_X = model.wv.vectors[node_vec_index]
                
                # 训练完后立即保存
                if self.graph_name is not None:
                    os.makedirs(self.graph_name, exist_ok=True)  # 确保目录存在
                    np.save(f'{self.graph_name}/{self.graph_name}_init_X.npy', init_X)
        else:
            # 随机初始化逻辑
            if self.graph_name is not None and os.path.exists(f'{self.graph_name}/{self.graph_name}_init_X_randn.npy'):
                init_X = np.load(f'{self.graph_name}/{self.graph_name}_init_X_randn.npy')
                print('读取随机初始化嵌入')
            else:
                print('生成随机初始化嵌入')
                init_X = np.random.randn(graph.number_of_nodes(), self.d * 2)
                
                if self.graph_name is not None:
                    os.makedirs(self.graph_name, exist_ok=True)
                    np.save(f'{self.graph_name}/{self.graph_name}_init_X_randn.npy', init_X)
        
        return init_X

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)