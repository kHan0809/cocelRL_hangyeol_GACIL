import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from Algorithm.GAC_origin import GAC
from torch.utils.tensorboard import SummaryWriter
from Common.Utils import set_seed
import time
import pickle

parser = argparse.ArgumentParser(description='PyTorch Generative Actor-Critic Args')
parser.add_argument('--env-name', default="Hopper-v2",help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--eval', type=bool, default=True,help='Evaluates a policy a policy every 10 episode (default: True)')

parser.add_argument('--seed', type=int, default=-1, metavar='N',help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',help='batch size (default: 256)')
parser.add_argument('--hidden_dim', type=int, default=(256,256), metavar='N',help='hidden size (default: 256)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', type=bool, default=True, help='run on CUDA (default: False)')
parser.add_argument('--log', default=True, type=bool, help='use tensorboard summary writer to log, if false, cannot use the features below')

#train parameter
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',help='target smoothing coefficient(τ) (default: 0.005) : polyak')
parser.add_argument('--lr', type=float, default=0.001, metavar='G',help='learning rate (default: 0.0003)')

parser.add_argument('--max_ep_len'     , type=int, default=1000, help='learning rate (default: 0.0003)')
parser.add_argument('--steps_per_epoch', type=int, default=4000, help='learning rate (default: 0.0003)')
parser.add_argument('--epochs',           type=int, default=250,   help='learning rate (default: 0.0003)')
parser.add_argument('--start_steps',     type=int, default=10000,   help='learning rate (default: 0.0003)')
parser.add_argument('--update_after',    type=int, default=1000,   help='learning rate (default: 0.0003)')
parser.add_argument('--update_every',    type=int, default=50,   help='learning rate (default: 0.0003)')

#test parameter
parser.add_argument('--num_test_episodes', type=int, default=1,   help='learning rate (default: 0.0003)')
parser.add_argument('--num_demo', type=int, default=50000,   help='learning rate (default: 0.0003)')

#GAC parameter
parser.add_argument('--expand_batch', type=int, default=100, help='learning rate (default: 0.0003)')
parser.add_argument('--alpha'     , type=float, default=1.0 ,help='MMD rate')
parser.add_argument('--alpha_min' , type=float, default=1.0)
parser.add_argument('--alpha_max' , type=float, default=1.8)
parser.add_argument('--beta'      , type=float, default=0.3)
parser.add_argument('--beta_start', type=float, default=0.3)
parser.add_argument('--beta_max'  , type=float, default=4.0)
parser.add_argument('--beta_step' , type=float, default=0.01)
parser.add_argument('--kernel'    ,   default='energy')

args = parser.parse_args()
seed = set_seed(args.seed)
print(seed)

# Environment
test_env = gym.make(args.env_name)
print('env & test:', args.env_name, 'is created!')

#Seed setting
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
test_env.seed(seed)

# Agent
print("obs_dim = {}, act_dim = {}".format(test_env.observation_space.shape[0], test_env.action_space.shape[0]))
act_limit = test_env.action_space.high[0]
agent = GAC(test_env.observation_space.shape[0], test_env.action_space.shape[0], args)
#pretrained_agent_load
agent.policy_target.load_state_dict(torch.load("./model_save/policy_Hopper-v2_250.pth"))
agent.critic_target.load_state_dict(torch.load("./model_save/critic_Hopper-v2_250.pth"))
#pretrained_obs_mean_std
agent.obs_mean = torch.Tensor(np.load("./model_save/Hopper-v2_obs_mean.npy"))
agent.obs_std  = torch.Tensor(np.load("./model_save/Hopper-v2_obs_std.npy"))

Demo = []
t = 1
test_avg_ret = 0.
while args.num_demo>t:
    test_state, done, test_ep_ret, test_ep_len = test_env.reset(), False, 0, 0
    while not (done or (test_ep_len == test_env._max_episode_steps)):
        # Take deterministic actions at test time
        # test_env.render()
        action = agent.select_action(test_state, evaluate=True)
        test_state, r, done, _ = test_env.step(action * act_limit)
        test_state_zfiltered = ((test_state - agent.obs_mean.numpy()) / (agent.obs_std.numpy() + 1e-8)).clip(-5.0,5.0)
        Demo.append(np.concatenate((test_state_zfiltered,action),axis=0).tolist())

        test_ep_ret += r
        test_ep_len += 1
        t += 1
        if t == args.num_demo+1:
            break

    test_avg_ret += test_ep_ret
test_avg_ret /= args.num_test_episodes
print(np.array(Demo).shape)
pickle.dump(Demo,open('expert_demo_hopper.p','wb'))
print("TestAvgEpRet : {:.2f}".format(test_avg_ret))
test_env.close()

