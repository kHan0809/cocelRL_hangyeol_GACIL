import torch
import torch.nn.functional as F
import numpy as np
from Common.Buffer import ReplayMemory, Buffer, ReplayBuffer
from Common.Utils import copy_weight, soft_update, hard_update
from torch.optim import Adam
from Model.Model import QNetwork, GenerativeGaussianMLPActor,  MLPQFunction_double, Discriminator
import math
import os
import time

class GACIL(object):
    def __init__(self, state_dim, action_dim, args):
        self.buffer_size = args.replay_size
        self.batch_size  = args.batch_size
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.gamma = args.gamma
        self.tau = args.tau

        self.expand_batch = args.expand_batch
        self.kernel = args.kernel
        self.kernel_routines = {"gaussian": self.gaussian_kernel, "energy": self.energy_kernel}


        self.critic = MLPQFunction_double(state_dim, action_dim, args.hidden_dim).to(device=self.device)
        self.q_optimizer = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = MLPQFunction_double(state_dim, action_dim, args.hidden_dim).to(device=self.device)
        hard_update(self.critic_target, self.critic)


        self.buffer = ReplayBuffer(obs_dim=state_dim, act_dim=action_dim, size=self.buffer_size)
        self.obs_std = torch.FloatTensor(self.buffer.obs_std).to(device=self.device)
        self.obs_mean = torch.FloatTensor(self.buffer.obs_mean).to(device=self.device)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        self.policy = GenerativeGaussianMLPActor(state_dim, action_dim,args.hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        self.policy_optim_GAN = Adam(self.policy.parameters(), lr=args.lr/100.0)
        self.policy_target = GenerativeGaussianMLPActor(state_dim, action_dim,args.hidden_dim).to(self.device)
        hard_update(self.policy_target, self.policy)

        for p in self.policy_target.parameters():
            p.requires_grad = False

        self.alpha_max=args.alpha_max
        self.alpha_min=args.alpha_min
        self.beta = args.beta
        self.beta_start=args.beta_start
        self.beta_step=args.beta_step
        self.beta_max =args.beta_max

        self.discrim = Discriminator(state_dim + action_dim, args.hidden_dim).to(self.device)
        self.discrim_optim = Adam(self.discrim.parameters(),lr=args.lr/100.0)



    def select_action(self, state, evaluate=False,state_limit=5.0):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        state = ((state - self.obs_mean.to(state.device)) / (self.obs_std.to(state.device) + 1e-8)).clamp(-state_limit, state_limit)
        with torch.no_grad():
            if evaluate is False:
                action = self.policy_target(state)
            else:
                action = self.policy_target(state,std=0.5)

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, batch_size):
        # Sample a batch from memory

        batch = self.buffer.sample_batch(batch_size=batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = batch['obs'], batch['act'], batch['rew'], batch['obs2'], batch['done']

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)


        with torch.no_grad():
            next_state_action = self.policy_target(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)



        #===critic update===
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()
        #====================

        #===reducing computation===
        for p in self.critic.parameters():
            p.requires_grad = False
        # ==========================

        #===Actor update===
        state_batch_repeat = state_batch.repeat(self.expand_batch,1)
        pi_repeat = self.policy(state_batch_repeat)

        qf1_pi, qf2_pi = self.critic(state_batch_repeat, pi_repeat)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)


        pi_repeat = pi_repeat.view(self.expand_batch, -1, pi_repeat.shape[-1]).transpose(0, 1)
        with torch.no_grad():
            uniform_action = (2*torch.rand_like(pi_repeat)-1)

        mmd_entropy = self.mmd(pi_repeat,uniform_action,kernel=self.kernel)

        loss_pi = -min_qf_pi.mean() + self.log_alpha.exp() * mmd_entropy

        self.policy_optim.zero_grad()
        loss_pi.backward()
        self.policy_optim.step()
        # ===================

        #===restore computation===
        for p in self.critic.parameters():
            p.requires_grad = True
        # ==========================
        #===auto alpha===
        self.alpha_optim.zero_grad()
        loss_log_alpha=self.compute_loss_log_alpha(mmd_entropy.detach().cpu().numpy(),self.beta)
        loss_log_alpha.backward()
        self.alpha_optim.step()

        with torch.no_grad():
            soft_update(self.critic_target, self.critic, self.tau)
            hard_update(self.policy_target, self.policy)


        return qf1.mean().item(), qf2.mean().item(), qf_loss.item(),  loss_pi.item(), loss_log_alpha.exp().item()

    def mmd(self, x, y, kernel='gaussian'):
        b = x.shape[0]
        m = x.shape[1]
        n = y.shape[1]

        if kernel in self.kernel_routines:
            kernel = self.kernel_routines[kernel]

        K_xx = kernel(x, x).mean()
        K_xy = kernel(x, y).mean()
        K_yy = kernel(y, y).mean()
        return self.sqrt_0(K_xx + K_yy - 2 * K_xy)

    def gaussian_kernel(self, x, y, blur=1.0):
        C2 = self.squared_distances(x / blur, y / blur)
        return (- 0.5 * C2).exp()

    def energy_kernel(self, x, y, blur=None):
        return -self.squared_distances(x, y)

    def squared_distances(self,x, y):
        if x.dim() == 2:
            D_xx = (x * x).sum(-1).unsqueeze(1)  # (N,1)
            D_xy = torch.matmul(x, y.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
            D_yy = (y * y).sum(-1).unsqueeze(0)  # (1,M)
        elif x.dim() == 3:  # Batch computation
            D_xx = (x * x).sum(-1).unsqueeze(2)  # (B,N,1)
            D_xy = torch.matmul(x, y.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
            D_yy = (y * y).sum(-1).unsqueeze(1)  # (B,1,M)
        else:
            print("x.shape : ", x.shape)
            raise ValueError("Incorrect number of dimensions")

        return D_xx - 2 * D_xy + D_yy

    def sqrt_0(self,x):
        return Sqrt0.apply(x)

    def compute_loss_log_alpha(self, mmd_entropy, beta):
        if self.log_alpha < -5.0:
            loss_log_alpha = -self.log_alpha
        elif self.log_alpha > 5.0:
            loss_log_alpha = self.log_alpha
        else:
            loss_log_alpha = self.log_alpha * (beta - mmd_entropy)
        return loss_log_alpha

    def change_beta(self,args):
        # Change beta.
        if self.log_alpha.exp() > args.alpha_max:
            self.beta += args.beta_step
            self.beta = min(self.beta, args.beta_max)
        elif self.log_alpha.exp() < args.alpha_min:
            self.beta -= args.beta_step
            self.beta = max(self.beta, args.beta_start)

    def stable_baseline(self):
        self.obs_std = torch.FloatTensor(self.buffer.obs_std).to(device=self.device)
        self.obs_mean = torch.FloatTensor(self.buffer.obs_mean).to(device=self.device)

    def get_reward(self,state,action):
        state = torch.Tensor(state).to(self.device)
        action = torch.Tensor(action).to(self.device)
        state_action = torch.cat([state,action])
        with torch.no_grad():
            return -math.log(self.discrim(state_action)[0].item())
    def train_discrim(self,demonstrations,batch_size):
        batch = self.buffer.sample_batch(batch_size=batch_size)
        state_batch = batch['obs']

        state_batch  = torch.FloatTensor(state_batch).to(self.device)
        action_batch = self.policy(state_batch)

        criterion = torch.nn.BCELoss()

        learner = self.discrim(torch.cat([state_batch, action_batch], dim=1))
        demonstrations = torch.Tensor(demonstrations).to(self.device)

        expert = self.discrim(demonstrations)

        discrim_loss = criterion(learner, torch.ones((state_batch.shape[0], 1)).to(self.device)) + criterion(expert, torch.zeros((demonstrations.shape[0], 1)).to(self.device))

        self.discrim_optim.zero_grad()
        discrim_loss.backward(retain_graph=True)
        self.discrim_optim.step()

        expert_acc = ((self.discrim(demonstrations) < 0.2).float()).mean()
        learner_acc = ((self.discrim(torch.cat([state_batch, action_batch], dim=1)) > 0.8).float()).mean()

        #==========train_G===============================

        learner = self.discrim(torch.cat([state_batch, action_batch], dim=1))
        G_loss  = criterion(learner, torch.ones((state_batch.shape[0], 1)).to(self.device))
        self.policy_optim_GAN.zero_grad()
        G_loss.backward()
        self.policy_optim_GAN.step()


        return  expert_acc, learner_acc


class Sqrt0(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        result = input.sqrt()
        result[input < 0] = 0
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        grad_input = grad_output / (2 * result)
        grad_input[result == 0] = 0
        return grad_input