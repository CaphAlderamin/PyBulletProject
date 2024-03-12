import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F

writer = SummaryWriter()
episode_steps = 0
def collect_trajectories(envs, policy, num_agents, action_size, tmax=200, nrand=5, seed=1, device=torch.device("cpu")):
    
    global episode_steps 
    global writer
    
    episode_rewards = 0
    
    # set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.backends.cudnn.deterministic = args.args.torch_deterministic
    
    def to_tensor(x, dtype=np.float32):
        return torch.from_numpy(np.array(x).astype(dtype)).to(device)
    
    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]
    value_list=[]
    done_list=[]

    state = envs.reset()

    # perform nrand random steps for exploration of env
    for _ in range(nrand):
        action = np.random.randn(num_agents, action_size)
        action = np.clip(action, -1.0, 1.0)
        _, reward, done, _   = envs.step(action)
        reward = torch.tensor(reward, device=device)
        
    for t in range(tmax):
        # Get the current states from the environment
        states = envs.get_screen().to(device)
        
        # Compute the estimated actions and their corresponding values
        action_est, values = policy(states)
        
        # Initialize the standard deviation parameter
        sigma = nn.Parameter(torch.zeros(action_size))
        sigma = F.softplus(sigma).to(device)
        
        # Sample actions from the Normal distribution
        dist = torch.distributions.Normal(action_est, sigma)
        actions = dist.sample()
        
        # Calculate the log probabilities of the sampled actions
        log_probs = dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=-1).detach()
        
        # Detach the values and actions from the computation graph
        values = values.detach()
        actions = actions.detach()
        
        # Step the environment using the sampled actions and store the results
        _, reward, done, _  = envs.step(actions.cpu().numpy())
        
        # Convert the rewards and dones to PyTorch tensors
        rewards = to_tensor(reward)
        dones = to_tensor(done)

        state_list.append(states.unsqueeze(0))
        prob_list.append(log_probs.unsqueeze(0))
        action_list.append(actions.unsqueeze(0))
        reward_list.append(rewards.unsqueeze(0))
        value_list.append(values.unsqueeze(0))
        done_list.append(dones)

        if np.any(dones.cpu().numpy()):
            episode_rewards += rewards.sum(dim=0)
            episode_steps += dones.sum(dim=0)
            
            # Log the average rewards for the episode
            avg_episode_rewards = episode_rewards.item() / dones.sum(dim=0).item()
            writer.add_scalar('Episodes average rewards', avg_episode_rewards, episode_steps.item())
            
            print(f"episode {episode_steps} done, rewards: {int(episode_rewards): <2} ({rewards.cpu().numpy().astype(int)})")
            
            # Reset the environment and reset the episode rewards and steps
            state = envs.reset()
            episode_rewards = 0
    
    # Convert the lists to PyTorch tensors            
    prob_list = torch.cat(prob_list, dim=0)
    state_list = torch.cat(state_list, dim=0)
    action_list = torch.cat(action_list, dim=0)
    reward_list = torch.cat(reward_list, dim=0)
    value_list = torch.cat(value_list, dim=0)
    done_list = torch.cat(done_list, dim=0)
    
    return prob_list, state_list, action_list, reward_list, value_list, done_list

def calc_returns(rewards, values, dones, device=torch.device("cpu")):
    num_step, num_agent = rewards.shape
    
    TAU = 0.95
    discount = 0.99
    
    # Create empty buffer
    #gae = torch.zeros(num_step, num_agent).float().to(device)
    #returns = torch.zeros(num_step, num_agent).float().to(device)
    gae = torch.zeros(num_step, num_agent, device=device)
    returns = torch.zeros(num_step, num_agent, device=device)

    # Set start values
    values_next = values[-1].detach()
    returns_current = values[-1].detach()
    
    gae_current = torch.zeros(num_agent).float().to(device)
    
    gammas = discount * (1. - dones.float())

    for irow in reversed(range(num_step)):
        values_current = values[irow]
        rewards_current = rewards[irow]
        gamma = gammas[irow]

        # Calculate TD Error
        td_error = rewards_current + gamma * values_next - values_current
        
        # Update GAE, returns
        gae_current = td_error + gamma * TAU * gae_current
        returns_current = rewards_current + gamma * returns_current
        
        # Set GAE, returns to buffer
        gae[irow] = gae_current
        returns[irow] = returns_current

        values_next = values_current

    return gae, returns

def get_screen(env, resize, device=torch.device("cpu")):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    #env.render(mode='human')
    screen = env._get_observation().transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

def eval_policy(envs, policy, action_size, resize, tmax=1000, device=torch.device("cpu")):
    reward_list=[]
    state = envs.reset()
    #states = torch.Tensor(envs.reset()[0]).to(device)
    for t in range(tmax):
        states = get_screen(envs, resize , device)
        action_est, values = policy(states)
        #action_est, _, _, _ = policy.get_action_and_value(states)
        sigma = nn.Parameter(torch.zeros(action_size))
        dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(device))
        actions = dist.sample()
        _, reward, done, *_  = envs.step(actions[0])
        #states, reward, done, *_  = envs.step(actions[0].cpu().numpy())
        dones = done
        reward_list.append(np.mean(reward))

        # stop if any of the trajectories is done to have retangular lists
        if np.any(dones):
            break
    return reward_list