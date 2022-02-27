import random
import numpy as np
import os
import pickle
import torch

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


class Buffer():

    def __init__(self, state_dim, action_dim, buffer_size,seed):
        random.seed(seed)
        self.states = np.zeros([buffer_size, state_dim], dtype=np.float32)
        self.actions = np.zeros([buffer_size, action_dim], dtype=np.float32)
        self.rewards = np.zeros([buffer_size, 1], dtype=np.float32)
        self.next_states = np.zeros([buffer_size, state_dim], dtype=np.float32)
        self.dones = np.zeros([buffer_size, 1], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, buffer_size

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs]
    def __len__(self):
        return self.size




class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, obs_limit=5.0):
        self.obs_buf = np.zeros(self.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(self.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        # For state normalization.
        self.total_num = 0
        self.obs_limit = 5.0
        self.obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_square_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_std = np.zeros(obs_dim, dtype=np.float32)

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        self.total_num += 1
        self.obs_mean = self.obs_mean / self.total_num * (self.total_num - 1) + np.array(obs) / self.total_num
        self.obs_square_mean = self.obs_square_mean / self.total_num * (self.total_num - 1) + np.array(obs) ** 2 / self.total_num
        self.obs_std = np.sqrt(self.obs_square_mean - self.obs_mean ** 2 + 1e-8)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_encoder(self.obs_buf[idxs]),
                     obs2=self.obs_encoder(self.obs2_buf[idxs]),
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def obs_encoder(self, o):
        return ((np.array(o) - self.obs_mean) / (self.obs_std + 1e-8)).clip(-self.obs_limit, self.obs_limit)

    def combined_shape(self, length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)