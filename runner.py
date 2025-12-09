import os
import pickle

import numpy as np
import torch
from tqdm import tqdm


# from MLBSN.environment import GraphEnvironment
from environment import GraphEnvironment


class Runner:
    def __init__(self, num_epochs, replay_buffer, batch_size, aggr_direction='all', use_deepwalk=True) -> None:
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.aggr_direction = aggr_direction
        self.use_deepwalk = use_deepwalk

    def run(self, env, agent,test_env, path):
        loss_list = []
        return_list = []
        # 测试环境
        # test_env = GraphEnvironment(env.graphs, env.locations, env.df_location, env.k, env.gamma, env.n_steps)
        # 先随机探索50条序列，获取训练样本
        self.explore(env, agent, 1, 1)
        eps_start = 1
        eps_end = 0.05
        eps_step = self.num_epochs // 2
        # 训练和测试
        # tbar = tqdm(total=num_epochs)

        for epoch in range(self.num_epochs):
            eps = eps_end + max(0, (eps_start - eps_end) * (eps_step - epoch) / eps_step)
            if eps == 0.5:
                break
            if epoch % 10 == 0:
                # 每10轮按照1-epsilon探索10条序列
                self.explore(env, agent, eps, 1)
                # 在原图上测试
                for k in [10]: # 2, 3, 4, 5, 10, 15, 20
                    test_env.k = k
                    rewards = self.explore(test_env, agent, 0, 1, False)

                return_list.append(rewards)
                # 保存q网络
                # if not self.use_deepwalk:
                #     path = f'qnet_ddqn_randn_init_x4'
                # else:
                #     # path = f'qnet_ddqn4'
                #     path = 'qnet_out'
                # print(path)
                torch.save(agent.q_net, f'{path}/q_net_{epoch}.model')
                # torch.save(agent.target_q_net, f'qnet_in/target_q_net_{epoch}.model')
            # 训练q网络
            states, actions, rewards, next_states, dones, graphs , i = self.replay_buffer.sample(self.batch_size)
            loss = agent.update(states, actions, rewards, next_states, graphs, dones)
            loss_list.append(loss)

        # with open(f'result-n/3k-5.txt', mode='w') as f:
        #     f.write(str(return_list))

    def explore(self, env, agent, eps, num_episodes, train=True):
        agent.epsilon = eps

        with tqdm(total=num_episodes, desc=f'epsilon={eps}时{"探索" if train else "测试"}{num_episodes}条序列') as bar:
            for i in range(num_episodes):
                state = env.reset()
                done = False
                episode_return = 0
                while not done:
                    action = agent.take_action(state, env)
                    reward, next_state, done = env.step(action)
                    episode_return += reward
                    state = next_state
                if train:
                    # 探索序列结束后计算n步奖励放经验回放池中
                    env.n_step_add_buffer(self.replay_buffer)
                    bar.set_postfix_str(f'buffer长度：{len(self.replay_buffer)}')
                else:
                    bar.set_postfix_str(f'测试图的种子集（{env.k}）：{env.seeds}，传播范围：{episode_return}。')
                bar.update(1)
            return episode_return
