import torch
import torch.nn as nn
import torch.nn.functional as F
from Common.Utils import weight_init, weight_init_Xavier
from torch.distributions import Normal

class GenerativeGaussianMLPActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.epsilon_dim = action_dim * action_dim
        self.hidden_dim = [hidden_dim[0], hidden_dim[1]]
        self.hidden_dim[0] += self.epsilon_dim

        self.network = nn.ModuleList([nn.Linear(state_dim+self.epsilon_dim, self.hidden_dim[0]), nn.ReLU()])
        for i in range(len(self.hidden_dim) - 1):
            self.network.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(self.hidden_dim[-1], action_dim))
        self.network.append(nn.Tanh())
        self.apply(weight_init_Xavier)

    def forward(self, state, std=1.0,noise='gaussian',epsilon_limit=5.0):
        if noise == 'gaussian':
            epsilon = (std * torch.randn(state.shape[0], self.epsilon_dim).to(device=state.device)).clamp(-epsilon_limit,epsilon_limit)
        else:
            epsilon = torch.rand(state.shape[0], self.epsilon_dim, device=state.device) * 2 - 1
        z = torch.cat([state, epsilon], dim=-1)
        for i in range(len(self.network)):
            z = self.network[i](z)
        return z


class GenerativeGaussianMLPActor2(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()
        self.epsilon_dim = act_dim * act_dim
        self.hidden_dim = [hidden_sizes[0], hidden_sizes[1]]
        self.hidden_dim[0] += self.epsilon_dim

        self.net = mlp([obs_dim+self.epsilon_dim] + list(self.hidden_dim) + [act_dim], activation, nn.Tanh())
        self.apply(weight_init_Xavier)

    def forward(self, obs, std=1.0, noise='gaussian', epsilon_limit=5.0):
        if noise == 'gaussian':
            epsilon = (std * torch.randn(obs.shape[0], self.epsilon_dim, device=obs.device)).clamp(-epsilon_limit, epsilon_limit)
        else:
            epsilon = torch.rand(obs.shape[0], self.epsilon_dim, device=obs.device) * 2 - 1
        pi_action = self.net(torch.cat([obs, epsilon], dim=-1))
        return pi_action


def mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    return nn.Sequential(*layers)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()
        self.hidden_dim = [hidden_sizes[0], hidden_sizes[1]]
        self.hidden_dim[0] += act_dim*act_dim

        self.q = mlp([obs_dim + act_dim] + list(self.hidden_dim) + [1], activation)
        self.apply(weight_init_Xavier)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return q # Critical to ensure q has right shape.


class MLPQFunction_double(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()
        self.hidden_dim = [hidden_sizes[0], hidden_sizes[1]]
        self.hidden_dim[0] += act_dim*act_dim

        self.q1 = mlp([obs_dim + act_dim] + list(self.hidden_dim) + [1], activation)
        self.q2 = mlp([obs_dim + act_dim] + list(self.hidden_dim) + [1], activation)
        self.apply(weight_init_Xavier)

    def forward(self, obs, act):
        q1 = self.q1(torch.cat([obs, act], dim=-1))
        q2 = self.q2(torch.cat([obs, act], dim=-1))

        return q1, q2 # Critical to ensure q has right shape.


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_sizes=(256,256)):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)

        self.apply(weight_init)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        prob = torch.sigmoid(self.fc3(x))
        return prob





class Squashed_Gaussian_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=(256, 256), log_std_min=-10, log_std_max=2):
        super(Squashed_Gaussian_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.network = nn.ModuleList([nn.Linear(state_dim, hidden_dim[0]), nn.ReLU()])
        for i in range(len(hidden_dim) - 1):
            self.network.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(hidden_dim[-1], action_dim * 2))

        self.apply(weight_init)

    def forward(self, state, deterministic=False):
        z = state
        for i in range(len(self.network)):
            z = self.network[i](z)

        mean, log_std = z.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=torch.diag_embed(std.pow(2),offset=0,dim1=-2,dim2=-1))

        if deterministic == True:
            tanh_mean = torch.tanh(mean)
            log_prob = dist.log_prob(mean)

            log_pi = log_prob.view(-1,1) - torch.log(1 - tanh_mean.pow(2) + 1e-6).sum(dim=1,keepdim=True)

            return tanh_mean, log_pi

        else:
            sample_action = dist.rsample()
            tanh_sample = torch.tanh(sample_action)
            log_prob = dist.log_prob(sample_action)

            log_pi = log_prob.view(-1,1) - torch.log(1 - tanh_sample.pow(2) + 1e-6).sum(dim=1, keepdim=True)
        return tanh_sample, log_pi


    def dist(self, state):
        z = state
        for i in range(len(self.network)):
            z = self.network[i](z)

        mean, log_std = z.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        return torch.distributions.Normal(mean, std)

    def mu_sigma(self, state):
        z = state
        for i in range(len(self.network)):
            z = self.network[i](z)

        mean, log_std = z.chunk(2, dim=-1)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        mean = torch.tanh(mean)

        return mean, std

    def entropy(self, state):
        dist = self.dist(state)
        return dist.entropy()


class Gaussian_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-10, log_std_max=2):
        super(Gaussian_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(self.state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, self.action_dim * 2)

        self.apply(weight_init)

    def forward(self, x, deterministic=False):
        L1     = F.relu(self.fc1(x))
        L2     = F.relu(self.fc2(L1))
        mean, log_std = self.fc3(L2).chunk(2, dim=-1)
        mean = torch.tanh(mean)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std, validate_args=True)

        if deterministic == True:
            log_prob = dist.log_prob(mean)
            return mean, log_prob

        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            # print("Forward action : ", action)
            return action, log_prob

    def dist(self, x):
        L1 = F.relu(self.fc1(x))
        L2 = F.relu(self.fc2(L1))
        mean, log_std = self.fc3(L2).chunk(2, dim=-1)
        mean = torch.tanh(mean)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std, validate_args=True)

        return dist

    def entropy(self, state):
        dist = self.dist(state)
        return dist.entropy()



class Policy_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, self.action_dim)

    def forward(self,x, activation = 'tanh'):
        L1 = F.relu(self.fc1(x))
        L2 = F.relu(self.fc2(L1))
        if activation == 'tanh':
            output = torch.tanh(self.fc3(L2))
        elif activation == 'softmax':
            output = torch.softmax(self.fc3(L2),dim=-1)
        else:
            output = self.fc3(L2)
        return output






class Actor(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, self.action_dim)

    def forward(self, x):
        L1 = F.relu(self.fc1(x))
        L2 = F.relu(self.fc2(L1))
        output = torch.tanh(self.fc3(L2))
        return output

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256):
        super(QNetwork, self).__init__()

        # Q1 architecture

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weight_init_Xavier)


    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class V_net(nn.Module):
    def __init__(self,state_dim):
        super(V_net, self).__init__()
        self.state_dim = state_dim

        self.fc1 = nn.Linear(self.state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x):
        L1 = F.relu(self.fc1(x))
        L2 = F.relu(self.fc2(L1))
        output = self.fc3(L2)

        return output

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)




class DeterministicPolicy(nn.Module):
    def __init__(self):
        pass
