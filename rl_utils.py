import collections
import pickle
import random
import time

import numpy as np
import networkx as nx
import torch


class Vocab:
    def __init__(self, location_count, sepc=['<pad>']):
        self.sepc = ['<unk>'] + sepc
        self.__unk = 0
        # 词频数的字典
        self.location_to_index = {w:i for i, w in enumerate(self.sepc)}
        self.location_to_index.update({l[0]:i for i, l in enumerate(sorted(location_count.items(), 
                                                                  key=lambda x:x[1], reverse=True), len(self.sepc))})
        # print(self.location_to_index)
        self.index_to_location = [l for l in self.location_to_index]
    
    def __getitem__(self, index):
        if isinstance(index, (list, tuple)):
            return [self.location_to_index.get(i, self.__unk) for i in index]
        return [self.location_to_index.get(index, self.__unk)]
    
    def __len__(self):
        return len(self.index_to_location)

    @property
    def unk(self):
        return self.__unk
    
# length 这个值，不同的图需要赋不一样的值
# 根据 length 这个截断词向量
def truncate_pad(location_list, vocab, length=600, pad='<pad>'):
    if len(location_list) >= length:
        return vocab[location_list[:length]]
    return vocab[location_list + vocab[pad] * (length - len(location_list))]


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done, graph, i):
        self.buffer.append((state, action, reward, next_state, done, graph, i))

    def sample(self, batch_size): 
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, graph, i = zip(*transitions)
        return state, action, reward, next_state, done, graph, i

    def size(self): 
        return len(self.buffer)
    
    def __len__(self): 
        return len(self.buffer)


# def get_edge_index(G):
#     edges = list(G.edges())
#     # 分离边的起点和终点
#     # src, dst = zip(*edges)
#     # print(edges)
#     # 转换为PyTorch张量
#     edge_index = np.array(edges).T
#     # edge_index = torch.tensor(edges, dtype=torch.long).T
#     edge_weights = torch.tensor(nx.to_numpy_array(G)[edge_index[0], edge_index[1]], dtype=torch.float)
#     edge_index = torch.tensor(edges, dtype=torch.long)
#     # print(edge_index)
#     return edge_index, edge_weights


def get_edge_index(G):
    # adj = nx.to_pandas_edgelist(G)
    # adj = adj.fillna(0)
    # source, target = adj['source'].to_numpy(), adj['target'].to_numpy()
    # edges = np.array([source, target])
    # # 转换为PyTorch张量
    # edge_index = torch.tensor(edges, dtype=torch.long)
    # edge_weights = torch.tensor(adj['p'].to_numpy(), dtype=torch.float)
    # print(adj[adj[['p']].isna()])
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).T
    edge_weights = torch.tensor([a['po'] * a['ps'] *a['ph'] for _, _, a in G.edges(data=True)])

    return edge_index, edge_weights


if __name__ == '__main__':
    # G = nx.erdos_renyi_graph(5, 0.5)
    # print(nx.to_numpy_array(G))

    with open(r'F:\program\LBSN\MLBSN\brightkite\G_20.pkl', 'rb') as f:
        G = pickle.load(f)
        t1 = time.time()
        get_edge_index(G)
        print(time.time()-t1)
        # print(get_edge_index(nx.to_numpy_array(G)))
    #     print(nx.to_numpy_array(G) != 0)
    
