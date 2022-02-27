import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from Algorithm.GACIL import GACIL
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
parser.add_argument('--num_test_episodes', type=int, default=10,   help='learning rate (default: 0.0003)')

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

#GAIL parameter
parser.add_argument('--discrim_update_num', type=int, default=50, help='update number of discriminator (default: 2)')
parser.add_argument('--suspend_accu_exp', type=float, default=0.80,help='accuracy for suspending discriminator about expert data (default: 0.8)')
parser.add_argument('--suspend_accu_gen', type=float, default=0.80,help='accuracy for suspending discriminator about generated data (default: 0.8)')
parser.add_argument('--lambda_gp', type=float, default=0.01, help='GP parameter')

args = parser.parse_args()
for iteration in range(2,6):
    # log
    if args.log == True:
        f = open("./log" + str(iteration) + "GACIL_v2" + ".txt", 'w')
        f.close()
    seed = set_seed(args.seed)
    print(seed)

    # Environment
    env, test_env = gym.make(args.env_name), gym.make(args.env_name)
    print('env & test:', args.env_name, 'is created!')

    #Seed setting
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env.seed(seed)
    # env.action_space.seed(seed)
    test_env.seed(seed)

    # Agent
    print("obs_dim = {}, act_dim = {}".format(env.observation_space.shape[0], env.action_space.shape[0]))
    act_limit = env.action_space.high[0]

    agent = GACIL(env.observation_space.shape[0], env.action_space.shape[0], args)
    #Expert-Demonstration & discrim
    expert_demo = pickle.load(open('./expert_demo_hopper.p', "rb"))
    demonstrations = np.array(expert_demo)
    train_discrim_flag = True


    # Training Loop
    total_steps = args.steps_per_epoch * args.epochs
    start_time = time.time()
    state, ep_ret, ep_len = env.reset(), 0, 0

    for t in range(total_steps):
        if t <= args.start_steps:
            action = env.action_space.sample() / act_limit
        else:
            action = agent.select_action(state)

        next_state, reward, done, _ = env.step(action * act_limit)  # Step
        irl_reward = agent.get_reward(state, action)

        ep_ret += reward
        ep_len += 1

        mask = 1.0 if ep_len == env._max_episode_steps else float(not done)
        agent.buffer.store(state, action, reward+irl_reward, next_state, mask)

        agent.stable_baseline()


        state = next_state

        # End of trajectory handling
        if done or (ep_len == env._max_episode_steps):
            # print("Total numsteps: {}, episode steps: {}, reward: {}".format(t, ep_len, round(ep_ret, 2)))
            state, ep_ret, ep_len = env.reset(), 0, 0


        # Update handling
        if t >= args.update_after and t % args.update_every == 0:
            epoch = (t + 1) // args.steps_per_epoch
            for j in range(args.update_every):
                critic1_value, critic2_value, critic_loss, pi_loss, alpha_value = agent.update_parameters(args.batch_size)

            #====Discriminator Train and Flag setting====
            if train_discrim_flag:
                expert_acc_mean, learner_acc_mean = 0, 0
                for i in range(args.discrim_update_num):
                    for m in range(10):
                        expert_acc, learner_acc = agent.train_discrim(demonstrations,args.batch_size)
                    agent.train_generator(args.batch_size)

                    expert_acc_mean  += expert_acc
                    learner_acc_mean += learner_acc
                expert_acc_mean /= args.discrim_update_num
                learner_acc_mean /= args.discrim_update_num

                print(reward, irl_reward)
                print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
                if expert_acc_mean > args.suspend_accu_exp and learner_acc_mean > args.suspend_accu_gen:
                    train_discrim_flag = False

            agent.change_beta(args)

        # End of epoch handling
        if (t+1) % args.steps_per_epoch == 0:
            epoch = (t+1) // args.steps_per_epoch

            #===========task_mean_std========
            # np.save("./model_save/obs_mean", agent.obs_mean.cpu().detach().numpy())
            # np.save("./model_save/obs_std", agent.obs_std.cpu().detach().numpy())
            # if epoch % 10 == 0:
            #     print("SAVE_model!")
            #     torch.save(agent.policy.state_dict(), "./model_save/policy_"+args.env_name+"_"+str(epoch)+".pth")
            #     torch.save(agent.critic.state_dict(), "./model_save/critic_"+args.env_name+"_"+str(epoch)+".pth")

            if epoch == args.epochs:
                pass
            #===========test============
            test_avg_ret = 0.
            for j in range(args.num_test_episodes):
                test_state, done, test_ep_ret, test_ep_len = test_env.reset(), False, 0, 0
                while not (done or (test_ep_len == env._max_episode_steps)):
                    # Take deterministic actions at test time
                    test_state, r, done, _ = test_env.step(agent.select_action(test_state, evaluate=True) * act_limit)
                    test_ep_ret += r
                    test_ep_len += 1
                test_avg_ret += test_ep_ret
            test_avg_ret /= args.num_test_episodes

            print("----------------------------------------")
            print("Epoch        : {}".format(epoch))
            print("TestAvgEpRet : {:.2f}".format(test_avg_ret))
            print("TestEplen    : {}".format(test_ep_len))
            print("Critic1      : {:.2f}".format(critic1_value))
            print("Critic2      : {:.2f}".format(critic2_value))
            print("Critic loss  : {:.2f}".format(critic_loss))
            print("Policy loss  : {:.2f}".format(pi_loss))
            print("Alpha        : {:.2f}".format(alpha_value))
            print("Beta         : {:.2f}".format(agent.beta))
            print("----------------------------------------")

            f = open("./log" + str(iteration) + "GACIL_v2" + ".txt", 'a')
            f.write(" ".join([str(t+1), str(int(round(test_avg_ret)))]))
            f.write("\n")
            f.close()

    env.close(), test_env.close()

