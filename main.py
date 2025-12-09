import os
import warnings
warnings.filterwarnings("ignore")
import pickle
import torch
import pandas as pd
import networkx as nx
import numpy as np
from runner import Runner
from environment import GraphEnvironment
from agent import Agent
from model import deep_walk
import rl_utils
import warnings
import random

def set_seed(seed):
    """设置所有随机数种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 为了确保可重复性，设置CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)

# 使用函数设置种子
set_seed(1)

if __name__ == '__main__':
    # 设置超参数
    lr = 0.001
    num_epochs = 1000
    gamma = 0.99
    epsilon = 0.01
    target_update = 100
    buffer_size = 50000
    batch_size = 16
    k = 5
    n_steps = 1
    num_features = 64
    # deepwalk的参数
    t = 10
    l = 20
    d = num_features
    w = 4
    epochs = 20 # skipgram训练轮数
    use_deepwalk = True
    aggr_direction = 'all'
    DDQN = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    with open('data_219/G_219.pkl', 'rb') as f:
        G = pickle.load(f)
    for u, v, a in G.edges(data=True):
        # print(a)
        a['po'] = a['p']
        a['ps'] = 1
        a['ph'] = 1
    # 读取子图
    graphs = []
    filenames = os.listdir('train_data')
    test_graphs = []
    for fn in filenames:
        graph = nx.read_edgelist(os.path.join('train_data', fn), nodetype=int, create_using=nx.DiGraph)
        for u, v, a in graph.edges(data=True):
            a['po'] = 1 / graph.in_degree(v)
            a['ps'] = random.uniform(0.001, 1)
            a['ph'] = random.uniform(0.001, 1)
        graphs.append(graph)
        print(graph)

    test_graphs = [G]
    # 环境、智能体
    env = GraphEnvironment(graphs, k, gamma, n_steps, num_workers=5)
    test_env = GraphEnvironment(test_graphs, k, gamma, n_steps, num_workers=5)
    agent = Agent(num_features // 2, gamma, epsilon, lr, device, target_update, n_steps, t, l, w, 
                    aggr_direction=aggr_direction, use_deepwalk=use_deepwalk, DDQN=DDQN)
    runner = Runner(num_epochs, replay_buffer, batch_size, use_deepwalk=use_deepwalk)

    if not use_deepwalk:
        path = f'randn_embedding_{aggr_direction}_{"DDQN" if DDQN else "DQN"}'
    else:
        path = f'{aggr_direction}_{"DDQN" if DDQN else "DQN"}'

    if not os.path.exists(path):
        os.mkdir(path)

    runner.run(env, agent, test_env, path)
