import argparse
import torch
import gym
import sys
import numpy as np
from Algorithm.PPO import PPO
from Common.Utils import set_seed, gym_env, Eval

sys.path.append('C:/Users/owner/.mujoco/mujoco200/bin')

def hyperparameters():
    parser = argparse.ArgumentParser(description='Soft Actor Critic (SAC) v2 example')
    #environment
    parser.add_argument('--env-name', default='Hopper-v2', help='Pendulum-v0, MountainCarContinuous-v0, CartPole-v0')
    parser.add_argument('--discrete', default=False, type=bool, help='whether the environment is discrete or not')
    parser.add_argument('--render', default=False, type=bool)


    parser.add_argument('--training-start', default=0, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=1000000, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=5000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=5, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=8463, type=int, help='Random seed setting')
    #PPO
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--buffer-size', default=30000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='offline', help='offline')
    parser.add_argument('--ppo-mode', default='clip', help='Clip, Adaptive KL, Fixed KL')
    parser.add_argument('--clip', default=0.2, type=float)
    parser.add_argument('--training-step', default=1, type=int, help='inverteddobulependulum-v2: 1')
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lambda-gae', default=1, type=float)
    parser.add_argument('--actor-lr', default=0.0003, type=float)
    parser.add_argument('--critic-lr', default=0.001, type=float)

    #log
    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')
    parser.add_argument('--log', default=False, type=bool, help='use tensorboard summary writer to log, if false, cannot use the features below')
    parser.add_argument('--tensorboard', default=True, type=bool, help='when logged, write in tensorboard')
    parser.add_argument('--file', default=False, type=bool, help='when logged, write log')

    parser.add_argument('--model', default=False, type=bool, help='when logged, save model')
    parser.add_argument('--model-freq', default=10000, type=int, help='model saving frequency')
    parser.add_argument('--buffer', default=False, type=bool, help='when logged, save buffer')
    parser.add_argument('--buffer-freq', default=10000, type=int, help='buffer saving frequency')

    args = parser.parse_args()

    return args

def main(args,iter):
    if args.cpu_only == True:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.log == True:
        f = open("./log" + str(iter) +"PPO"+".txt", 'w')
        f.close()

    print("Device: ", device)
    # random seed setting
    random_seed = set_seed(args.random_seed)
    print("Random Seed:", random_seed)

    env, eval_env = gym_env(args.env_name, random_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    episode_length = env._max_episode_steps

    reward_buff = []
    algorithm = PPO(state_dim,action_dim,device,args)

    state = env.reset()
    epi_reward = 0
    epi_timesteps = 0
    epi_num = 0
    cumul_step = 0

    for global_step in range(int(args.max_step)):
        action, log_prob = algorithm.get_action(state)

        next_state, reward, terminal, _ = env.step(action*max_action)
        epi_timesteps += 1

        done = float(terminal) if epi_timesteps < episode_length else 0

        algorithm.buffer.store(state, action, reward, next_state, done, log_prob)

        state = next_state
        epi_reward += reward


        if terminal & (global_step >= args.training_start):
            cumul_step += epi_timesteps
            if cumul_step>20480:
                print("Episode : ", epi_num, "Reward %0.2f" % epi_reward, "Local step : ", epi_timesteps,
                      "Global step : ", global_step)
                algorithm.train(args.training_step)
                cumul_step = 0

            state = env.reset()
            epi_reward = 0
            epi_timesteps = 0
            epi_num += 1



        # if ((global_step + 1) % args.eval_step==0)&(args.eval == True):
        #     min_rwd, avg_rwd, max_rwd = Eval(eval_env, algorithm, args.eval_episode,args.render)
        #     print(
        #         f"[#EPISODE {epi_num} | #GLOBALSTEP {global_step + 1}] MIN : {min_rwd:.2f}, AVE : {avg_rwd:.2f}, MAX : {max_rwd:.2f}")
        #
        #     # Logging (e.g., csv file / npy file/ txt file)
        #     global_step, min_rwd, avg_rwd, max_rwd = str(global_step + 1), str(min_rwd), str(avg_rwd), str(max_rwd)
        #
        #     f = open("./log" + str(iter) +"PPO"+".txt", 'a')
        #     f.write(" ".join([global_step, min_rwd, avg_rwd, max_rwd]))
        #     f.write("\n")
        #     f.close()



if __name__ == '__main__':
    for iter in range(1,2):
        args = hyperparameters()
        main(args,iter)