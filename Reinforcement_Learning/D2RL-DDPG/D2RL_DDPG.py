import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update , OrnsteinUhlenbeckNoise
from model import D2RL_Policy,D2RL_QNetwork
import random
import numpy as np

class D2RL_DDPG(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau

        self.device = torch.device("cuda")

        self.critic = D2RL_QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = D2RL_QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.policy=D2RL_Policy(num_inputs,action_space.shape[0],args.hidden_size,action_space).to(self.device)
        self.policy_optim=Adam(self.policy.parameters(),lr=args.lr)
        self.policy_target=D2RL_Policy(num_inputs,action_space.shape[0],args.hidden_size,action_space).to(self.device)
        hard_update(self.policy_target, self.policy)

        self.ou_noise=OrnsteinUhlenbeckNoise(mu=np.zeros(1))



    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action= self.policy(state)
            action=action.detach().cpu().numpy()[0]+self.ou_noise()[0]
        else:
            action = self.policy(state)
            action=action.detach().cpu().numpy()[0]
        return action

    def update_parameters(self, memory, batch_size, updates):

        s,a,r,s_prime,done_mask= memory.sample(batch_size=batch_size)

        s = torch.FloatTensor(s).to(self.device)
        s_prime = torch.FloatTensor(s_prime).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device).unsqueeze(1)
        done_mask = torch.FloatTensor(done_mask).to(self.device).unsqueeze(1)

        target = r + self.gamma * self.critic_target(s_prime, self.policy_target(s_prime)) * done_mask
        q_loss = F.smooth_l1_loss(self.critic(s, a), target.detach())
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        mu_loss = -self.critic(s, self.policy(s)).mean()
        self.policy_optim.zero_grad()
        mu_loss.backward()
        self.policy_optim.step()

        soft_update(self.critic_target, self.critic, self.tau)


    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()