import multiprocessing
import random
import statistics
import networkx as nx
import numpy as np
import multiprocessing
import statistics
from tqdm import tqdm
# import ndlib.models.ModelConfig as mc
# import ndlib.models.epidemics as ep


class GraphEnvironment:
    def __init__(self, graphs, k, gamma=0.99, n_steps=2, R=10000, num_workers=8):
        """
        G: networkx的图，Graph或DiGraph；
        locations: 
        k: 种子集大小；
        n_steps: 计算奖励时的步长；
        method: 计算奖励的方法；
        R: 使用蒙特卡洛估计奖励的轮数；
        num_workers: 使用多少个核心计算传播范围
        """
        self.graphs = graphs # 子图列表
        self.k = k
        self.gamma = gamma
        self.n_steps = n_steps
        self.R = R
        self.num_workers = num_workers
        self.graph = None # 当前使用的子图
        # 当前状态，每个位置表示一个节点是否被选择，1是已选，0是未选
        self.state = None
        # 前一状态的奖励
        self.preview_reward = 0
        # 记录每次探索的状态、动作、奖励、下一步状态，以便计算n步奖励（为了学习得更好，n步可以更好反应真实的情况）
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.seeds = []
        self.reward_records = {} # 记录种子集的奖励

    def reset(self):
        """
        重置环境。
        """
        self.graph = random.choice(self.graphs) # 随机选一个子图
        self.seeds = []
        self.state = [0] * self.graph.number_of_nodes()
        self.preview_reward = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        return self.state
    
    def step(self, action):
        """
        根据所给的action，转移到新状态。
        """
        self.states.append(self.state.copy())
        self.state[action] = 1  # 更新状态
        self.seeds.append(action)
        # 计算奖励
        reward = self.compute_reward()

        done = False
        if len(self.seeds) == self.k:
            done = True

        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(self.state)
        return reward, self.state, done

    def IC(self, S, mc):
        graph = self.graph.copy()
        num = 0
        for i in range(mc):
            new_active, A = S[:], S[:]
            while new_active:
                new_ones = []
                for node in new_active:
                    for s in graph.neighbors(node):
                        if s not in S:
                            p = graph.edges[node, s]['po'] * graph.edges[node, s]['ps'] * graph.edges[node, s]['ph']
                            if np.random.uniform(0, 1) < p:
                                new_ones.append(s)
                                
                new_active = list(set(new_ones) - set(A))
                A += new_active
            num += len(A)
        return num / mc

    def computeLT(self, seeds, R=10000):
        num = 0

        for _ in range(R):
            model = ep.ThresholdModel(self.graph)

            config = mc.Configuration()
            config.add_model_initial_configuration('Infected', seeds)

            for i in self.graph.nodes():
                config.add_node_configuration("threshold", i, random.random())

            model.set_initial_status(config)
            num += model.iteration_bunch(20, node_status=False)[-1]['node_count'][1]

        return num / R

    def compute_reward(self):
        if f'{str(id(self.graph))}{sorted(self.seeds)}' in self.reward_records:
            r = self.reward_records[f'{str(id(self.graph))}{sorted(self.seeds)}']
        else:
            with multiprocessing.Pool(self.num_workers) as pool:
                args = [[self.seeds, int(self.R/self.num_workers)] for _ in range(self.num_workers)]
                results = pool.starmap(self.IC, args)
                # results = pool.starmap(self.computeLT, args)
            r = statistics.mean(results)
            self.reward_records[f'{str(id(self.graph))}{sorted(self.seeds)}'] = r
        reward = r - self.preview_reward
        self.preview_reward = r
        return reward


    def n_step_add_buffer(self, buffer):
        """
        计算n步奖励，并放入经验回放池中。
        buffer: 经验回放池。
        """
        for i in range(len(self.states)):
            if i + self.n_steps < len(self.states):
                # 此时才能计算n步累加折扣奖励
                n_reward = 0
                for r in self.rewards[i:i+self.n_steps][::-1]:
                    # 倒序
                    n_reward += r/self.graph.number_of_nodes() + self.gamma * n_reward
                buffer.add(self.states[i], self.actions[i], n_reward, self.states[i+self.n_steps],
                           False, self.graph, self.graphs.index(self.graph))
            elif i + self.n_steps == len(self.states):
                # 最后一个状态是终止状态
                n_reward = 0
                for r in self.rewards[i:][::-1]:
                    # 倒序
                    n_reward += r/self.graph.number_of_nodes() + self.gamma * n_reward
                buffer.add(self.states[i], self.actions[i], n_reward, self.states[-1],
                           True, self.graph, self.graphs.index(self.graph))
