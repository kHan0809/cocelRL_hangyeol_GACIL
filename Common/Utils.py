import numpy as np
import gym
import random
import torch
import torch.nn as nn


def Eval(eval_env, agent, eval_num,render):
    reward_history = []
    max_action = float(eval_env.action_space.high[0])
    for eval_iter in range(eval_num):
        state = eval_env.reset()
        reward_sum = 0
        for eval_step in range(eval_env._max_episode_steps):
            if (eval_iter == eval_num-1)&(render):
                eval_env.render()
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminal, _ = eval_env.step(action*max_action)
            reward_sum += reward
            state = next_state
            if terminal:
                break

        reward_history.append(reward_sum)
    return min(reward_history), sum(reward_history)/len(reward_history), max(reward_history)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def copy_weight(network, target_network):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(param.data)

def set_seed(random_seed):
    if random_seed <= 0:
        random_seed = np.random.randint(1, 9999)
    else:
        random_seed = random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    random.seed(random_seed)

    return random_seed


def gym_env(env_name, random_seed):
    import gym
    # openai gym
    env = gym.make(env_name)
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    test_env = gym.make(env_name)
    test_env.seed(random_seed)
    test_env.action_space.seed(random_seed)

    return env, test_env

def weight_init_Xavier(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight, gain=0.01)
        module.bias.data.zero_()

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers.
        Reference: https://github.com/MishaLaskin/rad/blob/master/curl_sac.py"""

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

