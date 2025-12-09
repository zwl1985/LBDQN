import multiprocessing
import os
import statistics
import time
import warnings
warnings.filterwarnings("ignore")
import pickle
import torch
from environment import GraphEnvironment
from agent import Agent

import argparse
import sys

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gn', '--graph_name',
                       type=str,
                       default='brightkite',
                       help='图名称')
    
    parser.add_argument('-ad', '--aggr_direction',
                       type=str,
                       default='all',
                       help='聚合方向')
    
    parser.add_argument('-fn', '--folder_name',
                       type=str,
                       default='pth/lbdqn.model',
                       help='模型参数路径')
    
    parser.add_argument('-lc', '--location_counts',
                       type=int,
                       default='1',
                       help='地点个数（1，5，10，15，20）')
    
    parser.add_argument('-p', '--processes',
                       type=int,
                       default=4,
                       help='进程数 (默认: 4)')
    
    return parser


if __name__ == '__main__':
    n_steps = 1
    num_features = 64
    use_deepwalk =True

    parser = create_parser()
    args = parser.parse_args()
    print(args)
    graph_name = args.graph_name
    folder_name = args.folder_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(f'{graph_name}/g_{args.location_counts}.pkl', 'rb') as f:
        G = pickle.load(f)

    print(G)
    test_graphs = [G]
    # 环境、智能体
    test_env = GraphEnvironment(test_graphs, test_graphs.copy(), 0, 0, 0, num_workers=1)
    aggr_direction = args.aggr_direction
    agent = Agent(num_features // 2,
                  0, 0, 0, device, 0, n_steps, aggr_direction=aggr_direction, graph_name=graph_name, use_deepwalk=use_deepwalk)
    processes = args.processes

    agent.q_net.load_state_dict(torch.load(folder_name).state_dict())  
    for k in range(50, 301, 50):
        t1 = time.time()
        data = agent.get_q_net_input(test_graphs)
        states = [test_env.reset()]
        agent.q_net.eval()
        res = agent.q_net(data.x, data.edge_index, data.edge_weights, data.batch, states)
    
        seeds = torch.topk(res, k=k)[1].tolist()

        with multiprocessing.Pool(processes) as pool:
            args = [[seeds, int(10000 / processes)] for _ in range(processes)]
            results = pool.starmap(test_env.IC, args)

        r = statistics.mean(results)

        t2 = time.time()
        print(k, r)

