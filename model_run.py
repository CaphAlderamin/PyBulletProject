import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F

writer = SummaryWriter()
i_episode = 0
def collect_trajectories(envs, policy, num_agents, action_size, tmax=200, nrand=5, device=torch.device("cpu")):
    
    global i_episode 
    global writer
    
    episode_rewards = 0
    
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

    # perform nrand random steps
    for _ in range(nrand):
        action = np.random.randn(num_agents, action_size)
        action = np.clip(action, -1.0, 1.0)
        _, reward, done, _   = envs.step(action)
        reward = torch.tensor(reward, device=device)
        
    for t in range(tmax):
        states = envs.get_screen().to(device)
        action_est, values = policy(states)
        sigma = nn.Parameter(torch.zeros(action_size))
        #sigma = nn.Parameter(torch.zeros(action_size).to(device))
        dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(device))
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=-1).detach()
        values = values.detach()
        actions = actions.detach()
        
        _, reward, done, _  = envs.step(actions.cpu().numpy())
        rewards = to_tensor(reward)
        dones = to_tensor(done)

        #print(f"rewards: {rewards}")
        #print(f"dones: {dones}")

        state_list.append(states.unsqueeze(0))
        prob_list.append(log_probs.unsqueeze(0))
        action_list.append(actions.unsqueeze(0))
        reward_list.append(rewards.unsqueeze(0))
        value_list.append(values.unsqueeze(0))
        done_list.append(dones)

        if np.any(dones.cpu().numpy()):
            episode_rewards += rewards.sum(dim=0)
            i_episode += dones.sum(dim=0)
            writer.add_scalar('Episodes average rewards', episode_rewards.item()/dones.sum(dim=0).item(), i_episode.item())
            print(f"episode {i_episode} done, rewards: {int(episode_rewards): <2} ({rewards.cpu().numpy().astype(int)})")
            state = envs.reset()
            episode_rewards = 0
                
    state_list = torch.cat(state_list, dim=0)
    prob_list = torch.cat(prob_list, dim=0)
    action_list = torch.cat(action_list, dim=0)
    reward_list = torch.cat(reward_list, dim=0)
    value_list = torch.cat(value_list, dim=0)
    done_list = torch.cat(done_list, dim=0)
    
    return prob_list, state_list, action_list, reward_list, value_list, done_list

def calc_returns(rewards, values, dones, device=torch.device("cpu")):
    n_step = len(rewards)
    n_agent = len(rewards[0])

    # Create empty buffer
    GAE = torch.zeros(n_step,n_agent).float().to(device)
    returns = torch.zeros(n_step,n_agent).float().to(device)

    # Set start values
    GAE_current = torch.zeros(n_agent).float().to(device)

    TAU = 0.95
    discount = 0.99
    values_next = values[-1].detach()
    returns_current = values[-1].detach()
    for irow in reversed(range(n_step)):
        values_current = values[irow]
        rewards_current = rewards[irow]
        gamma = discount * (1. - dones[irow].float())

        # Calculate TD Error
        td_error = rewards_current + gamma * values_next - values_current
        # Update GAE, returns
        GAE_current = td_error + gamma * TAU * GAE_current
        returns_current = rewards_current + gamma * returns_current
        # Set GAE, returns to buffer
        GAE[irow] = GAE_current
        returns[irow] = returns_current

        values_next = values_current

    return GAE, returns

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