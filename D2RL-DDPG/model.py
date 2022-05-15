import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, 1)



        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1= F.relu(self.linear3(x1))
        x1= F.relu(self.linear4(x1))
        x1 = self.linear5(x1)

        return x1


class D2RL_QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(D2RL_QNetwork, self).__init__()

        in_dim=num_inputs+num_actions+hidden_dim

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(in_dim, hidden_dim)
        self.linear3 = nn.Linear(in_dim, hidden_dim)
        self.linear4 = nn.Linear(in_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, 1)


        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1=torch.cat([x1,xu],dim=1)
        x1 = F.relu(self.linear2(x1))
        x1 = torch.cat([x1, xu], dim=1)
        x1= F.relu(self.linear3(x1))
        x1 = torch.cat([x1, xu], dim=1)
        x1= F.relu(self.linear4(x1))
        x1 = self.linear5(x1)


        return x1

class Policy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3=nn.Linear(hidden_dim,hidden_dim)
        self.linear4=nn.Linear(hidden_dim,hidden_dim)

        self.action = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        self.boundary=torch.tensor([action_space.high])

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x=F.relu(self.linear3(x))
        x=F.relu(self.linear4(x))

        action=torch.tanh(self.action(x))*self.boundary
        return action


    def to(self, device):
        self.boundary = self.boundary.to(device)
        return super(Policy, self).to(device)


class D2RL_Policy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(D2RL_Policy, self).__init__()
        in_dim=hidden_dim+num_inputs

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(in_dim, hidden_dim)
        self.linear3=nn.Linear(in_dim,hidden_dim)
        self.linear4=nn.Linear(in_dim,hidden_dim)


        self.action=nn.Linear(hidden_dim,1)
        self.boundary=torch.tensor([action_space.high])
        self.apply(weights_init_)

    def forward(self, state):

        x = F.relu(self.linear1(state))
        x=torch.cat([x,state],dim=1)
        x = F.relu(self.linear2(x))
        x = torch.cat([x, state], dim=1)
        x=F.relu(self.linear3(x))
        x = torch.cat([x, state], dim=1)
        x=F.relu(self.linear4(x))

        action=torch.tanh(self.action(x))*self.boundary
        return action


    def to(self, device):
        self.boundary=self.boundary.to(device)
        return super(D2RL_Policy, self).to(device)

