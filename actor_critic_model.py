import torch
import torch.nn as nn
import torch.nn.functional as F

def build_hidden_layer(input_dim, hidden_layers):
    """Build hidden layer.
    Params
    ======
        input_dim (int): Dimension of hidden layer input
        hidden_layers (list(int)): Dimension of hidden layers
    """
    hidden = nn.ModuleList([nn.Linear(input_dim, hidden_layers[0])])
    if len(hidden_layers)>1:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        hidden.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
    return hidden

class MultiheadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionLayer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        attn_output, _ = self.multihead_attention(x, x, x)
        attn_output = self.layer_norm(attn_output + x)
        ff_output = self.feed_forward(attn_output)
        return self.layer_norm(ff_output + attn_output)

class ActorCritic(nn.Module):
    def __init__(self,channels,state_size,action_size,shared_layers,
                 critic_hidden_layers=[],actor_hidden_layers=[],
                 nhead=8, seed=1, init_type=None):
        """Initialize parameters and build policy.
        Params
        ======
            state_size (int,int,int): Dimension of each state
            action_size (int): Dimension of each action
            shared_layers (list(int)): Dimension of the shared hidden layers
            critic_hidden_layers (list(int)): Dimension of the critic's hidden layers
            actor_hidden_layers (list(int)): Dimension of the actor's hidden layers
            seed (int): Random seed
            init_type (str): Initialization type
        """
        super(ActorCritic, self).__init__()
        
        self.init_type = init_type
        self.seed = torch.manual_seed(seed)
        #self.sigma = nn.Parameter(torch.zeros(action_size))

        # Add shared hidden layer
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        # for larger layers
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size[0])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size[1])))
        linear_input_size = convh * convw * 64
        
        # Add first MultiheadAttention layer after conv layers
        self.attention_layer1 = MultiheadAttentionLayer(embed_dim=linear_input_size, num_heads=nhead)
        
        self.shared_layers = build_hidden_layer(input_dim=linear_input_size,
                                                hidden_layers=shared_layers)

        ## Add second MultiheadAttention layer before splitting into actor and critic
        #self.attention_layer2 = MultiheadAttentionLayer(embed_dim=shared_layers[-1], num_heads=nhead)
        
        # Add critic layers
        if critic_hidden_layers:
            # Add hidden layers for critic net if critic_hidden_layers is not empty
            self.critic_hidden = build_hidden_layer(input_dim=shared_layers[-1],
                                                    hidden_layers=critic_hidden_layers)
            self.critic = nn.Linear(critic_hidden_layers[-1], 1)
        else:
            self.critic_hidden = None
            self.critic = nn.Linear(shared_layers[-1], 1)

        # Add actor layers
        if actor_hidden_layers:
            # Add hidden layers for actor net if actor_hidden_layers is not empty
            self.actor_hidden = build_hidden_layer(input_dim=shared_layers[-1],
                                                   hidden_layers=actor_hidden_layers)
            self.actor = nn.Linear(actor_hidden_layers[-1], action_size)
        else:
            self.actor_hidden = None
            self.actor = nn.Linear(shared_layers[-1], action_size)

        # Apply Tanh() to bound the actions
        self.tanh = nn.Tanh()

        # Initialize hidden and actor-critic layers
        if self.init_type is not None:
            self.shared_layers.apply(self._initialize)
            self.critic.apply(self._initialize)
            self.actor.apply(self._initialize)
            if self.critic_hidden is not None:
                self.critic_hidden.apply(self._initialize)
            if self.actor_hidden is not None:
                self.actor_hidden.apply(self._initialize)

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

    def forward(self, state):
        """Build a network that maps state -> (action, value).
        """
        def apply_multi_layer(layers, x, f=F.leaky_relu):
            for layer in layers:
                x = f(layer(x))
            return x

        # Apply convolutional layer
        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = F.relu(self.bn3(self.conv3(state)))
        
        # Apply flatten layer
        state = self.flatten(state)
        
        # Apply first attention layer
        state = state.unsqueeze(1)
        state = self.attention_layer1(state)
        state = state.squeeze(1)
        
        state = apply_multi_layer(self.shared_layers, state.view(state.size(0),-1))

        # Apply second attention layer
        #state = state.unsqueeze(1)
        #state = self.attention_layer2(state)
        #state = state.squeeze(1)
        
        v_hid = state
        if self.critic_hidden is not None:
            v_hid = apply_multi_layer(self.critic_hidden,v_hid)

        a_hid = state
        if self.actor_hidden is not None:
            a_hid = apply_multi_layer(self.actor_hidden,a_hid)

        action = self.tanh(self.actor(a_hid))
        value = self.critic(v_hid).squeeze(-1)
        return action, value
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)