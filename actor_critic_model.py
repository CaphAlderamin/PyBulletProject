import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, envs, observation_size, init_type=None):
        super(ActorCritic, self).__init__()
        
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(observation_size[0], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(observation_size[1], 8, 4), 4, 2), 3, 1)
        linear_input_size = convh * convw * 64
        
        self.init_type = init_type
        
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 16, kernel_size=8, stride=4)),
            #nn.BatchNorm2d(32), 5
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=4, stride=2)),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1)),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            
            #layer_init(nn.Linear(64 * 2 * 2, 512)),
            layer_init(nn.Linear(linear_input_size, 512)),
            nn.ReLU()
            
            #layer_init(nn.Linear(linear_input_size, 512)),
            #nn.LeakyReLU(),
            #layer_init(nn.Linear(512, 256)),
            #nn.LeakyReLU(),
            #layer_init(nn.Linear(256, 128)),
            #nn.LeakyReLU(),
            #layer_init(nn.Linear(128, 64)),
            #nn.LeakyReLU(),
        )
        
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.actor_mean = layer_init(nn.Linear(512, np.prod(envs.action_space.shape)), std=0.01)
        
        #self.critic = nn.Sequential(
        #    layer_init(nn.Linear(64, 512)),
        #    nn.LeakyReLU(),
        #    layer_init(nn.Linear(512, 1), std=1),
        #    nn.LeakyReLU(),
        #)
        #self.actor_mean = nn.Sequential(
        #    layer_init(nn.Linear(64, 512)),
        #    nn.LeakyReLU(),
        #    layer_init(nn.Linear(512, np.prod(envs.action_space.shape)), std=0.01),
        #    nn.LeakyReLU(),
        #)
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

        if self.init_type is not None:
            self.network.apply(self._initialize)
            self.critic.apply(self._initialize)
            self.actor_mean.apply(self._initialize)
    
    def _initialize(self, n):
        """Initialize network weights.
        """
        if isinstance(n, nn.Linear):
            if self.init_type=='xavier-uniform':
                nn.init.xavier_uniform_(n.weight.data)
            elif self.init_type=='xavier-normal':
                nn.init.xavier_normal_(n.weight.data)
            elif self.init_type=='kaiming-uniform':
                nn.init.kaiming_uniform_(n.weight.data)
            elif self.init_type=='kaiming-normal':
                nn.init.kaiming_normal_(n.weight.data)
            elif self.init_type=='orthogonal':
                nn.init.orthogonal_(n.weight.data)
            elif self.init_type=='uniform':
                nn.init.uniform_(n.weight.data)
            elif self.init_type=='normal':
                nn.init.normal_(n.weight.data)
            else:
                raise KeyError(f'initialization type {self.init_type} not found')
    
    def get_value(self, x):
        #return self.critic(self.network(x / 255))
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        #print(f"==============x: {x}")
        #print(f"==============x: {x.shape}")
        #hidden = self.network(x / 255)
        hidden = self.network(x)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        #print(f"action.shape {action.shape}")    
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden)

    