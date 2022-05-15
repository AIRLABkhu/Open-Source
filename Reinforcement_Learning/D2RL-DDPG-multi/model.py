import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_layers):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.l1_1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.l1_2 = nn.Linear(hidden_dim, hidden_dim)


        if num_layers > 2:
            self.l1_3 = nn.Linear(hidden_dim, hidden_dim)
            self.l1_4 = nn.Linear(hidden_dim, hidden_dim)



        if num_layers > 4:
            self.l1_5 = nn.Linear(hidden_dim, hidden_dim)
            self.l1_6 = nn.Linear(hidden_dim, hidden_dim)


        if num_layers == 8:
            self.l1_7 = nn.Linear(hidden_dim, hidden_dim)
            self.l1_8 = nn.Linear(hidden_dim, hidden_dim)


        self.out1 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

        self.num_layers = num_layers

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.l1_1(xu))
        x1 = F.relu(self.l1_2(x1))


        if self.num_layers > 2:
            x1 = F.relu(self.l1_3(x1))
            x1 = F.relu(self.l1_4(x1))

        if self.num_layers > 4:
            x1 = F.relu(self.l1_5(x1))
            x1 = F.relu(self.l1_6(x1))


        if self.num_layers == 8:
            x1 = F.relu(self.l1_7(x1))
            x1 = F.relu(self.l1_8(x1))

        x1 = self.out1(x1)

        return x1


class D2RL_QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_layers):
        super(D2RL_QNetwork, self).__init__()

        in_dim = num_inputs + num_actions + hidden_dim
        # Q1 architecture
        self.l1_1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.l1_2 = nn.Linear(in_dim, hidden_dim)


        if num_layers > 2:
            self.l1_3 = nn.Linear(in_dim, hidden_dim)
            self.l1_4 = nn.Linear(in_dim, hidden_dim)



        if num_layers > 4:
            self.l1_5 = nn.Linear(in_dim, hidden_dim)
            self.l1_6 = nn.Linear(in_dim, hidden_dim)



        if num_layers == 8:
            self.l1_7 = nn.Linear(in_dim, hidden_dim)
            self.l1_8 = nn.Linear(in_dim, hidden_dim)


        self.out1 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

        self.num_layers = num_layers

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.l1_1(xu))
        x1 = torch.cat([x1, xu], dim=1)

        x1 = F.relu(self.l1_2(x1))
        if not self.num_layers == 2:
            x1 = torch.cat([x1, xu], dim=1)

        if self.num_layers > 2:
            x1 = F.relu(self.l1_3(x1))
            x1 = torch.cat([x1, xu], dim=1)

            x1 = F.relu(self.l1_4(x1))
            if not self.num_layers == 4:
                x1 = torch.cat([x1, xu], dim=1)

        if self.num_layers > 4:
            x1 = F.relu(self.l1_5(x1))
            x1 = torch.cat([x1, xu], dim=1)

            x1 = F.relu(self.l1_6(x1))
            if not self.num_layers == 6:
                x1 = torch.cat([x1, xu], dim=1)

        if self.num_layers == 8:
            x1 = F.relu(self.l1_7(x1))
            x1 = torch.cat([x1, xu], dim=1)

            x1 = F.relu(self.l1_8(x1))

        x1 = self.out1(x1)

        return x1

class Policy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_layers,action_space=None):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        if num_layers > 2:
            self.linear3 = nn.Linear(hidden_dim, hidden_dim)
            self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        if num_layers > 4:
            self.linear5 = nn.Linear(hidden_dim, hidden_dim)
            self.linear6 = nn.Linear(hidden_dim, hidden_dim)
        if num_layers == 8:
            self.linear7 = nn.Linear(hidden_dim, hidden_dim)
            self.linear8 = nn.Linear(hidden_dim, hidden_dim)

        self.action=nn.Linear(hidden_dim,1)
        self.boundary=torch.tensor([action_space.high])
        self.apply(weights_init_)
        self.num_layers=num_layers
    def forward(self, state):

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        if self.num_layers > 2:
            x = F.relu(self.linear3(x))
            x = F.relu(self.linear4(x))

        if self.num_layers > 4:
            x = F.relu(self.linear5(x))
            x = F.relu(self.linear6(x))

        if self.num_layers == 8:
            x = F.relu(self.linear7(x))
            x = F.relu(self.linear8(x))

        action=torch.tanh(self.action(x))*self.boundary
        return action


    def to(self, device):
        self.boundary=self.boundary.to(device)
        return super(Policy, self).to(device)


class D2RL_Policy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_layers,action_space=None):
        super(D2RL_Policy, self).__init__()
        in_dim=hidden_dim+num_inputs

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 2:
            self.linear3 = nn.Linear(in_dim, hidden_dim)
            self.linear4 = nn.Linear(in_dim, hidden_dim)
        if num_layers > 4:
            self.linear5 = nn.Linear(in_dim, hidden_dim)
            self.linear6 = nn.Linear(in_dim, hidden_dim)
        if num_layers == 8:
            self.linear7 = nn.Linear(in_dim, hidden_dim)
            self.linear8 = nn.Linear(in_dim, hidden_dim)

        self.action=nn.Linear(hidden_dim,1)
        self.boundary=torch.tensor([action_space.high])
        self.apply(weights_init_)
        self.num_layers=num_layers

    def forward(self, state):

        x = F.relu(self.linear1(state))
        x = torch.cat([x, state], dim=1)

        x = F.relu(self.linear2(x))

        if self.num_layers > 2:
            x = torch.cat([x, state], dim=1)
            x = F.relu(self.linear3(x))

            x = torch.cat([x, state], dim=1)
            x = F.relu(self.linear4(x))

        if self.num_layers > 4:
            x = torch.cat([x, state], dim=1)
            x = F.relu(self.linear5(x))

            x = torch.cat([x, state], dim=1)
            x = F.relu(self.linear6(x))

        if self.num_layers == 8:
            x = torch.cat([x, state], dim=1)
            x = F.relu(self.linear7(x))

            x = torch.cat([x, state], dim=1)
            x = F.relu(self.linear8(x))

        action=torch.tanh(self.action(x))*self.boundary
        return action


    def to(self, device):
        self.boundary=self.boundary.to(device)
        return super(D2RL_Policy, self).to(device)

