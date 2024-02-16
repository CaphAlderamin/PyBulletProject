import numpy as np

import torch
from gym import spaces
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv  
import pybullet as p

from Attendants import select_device
from ActorCriticModel import *
from ModelRun import *

if __name__ == "__main__":
    # Select a device for PyTorch calculations
    device = select_device(info=True)
    # PATH to evaluate model
    #PATH = 'models/policy_ppo_m50_normal2.pt'
    PATH = 'models\policy_ppo_m50_big1.pt'
    # Evaluate episodes
    episodes = 50
    
    env = KukaDiverseObjectEnv(
            renders=False, 
            isDiscrete=False, 
            removeHeightHack=False, 
            maxSteps=20, 
            isTest=True
        )
    env.observation_space = spaces.Box(low=0., high=1., shape=(84, 84, 3), dtype=np.float32)
    env.action_space = spaces.Box(low=-1, high=1, shape=(5,1))
    env.reset()
    
    # Size of each action
    action_size = env.action_space.shape[0]
    print('Size of each action:', action_size)
    # Size of screen
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape
    
    #policy=ActorCritic(
    #        state_size=(screen_height, screen_width),
    #        action_size=action_size,
    #        shared_layers=[128, 64],
    #        critic_hidden_layers=[64],
    #        actor_hidden_layers=[64],
    #        init_type='xavier-uniform',
    #        seed=0
    #    ).to(device)
    policy=ActorCritic(
            state_size=(screen_height, screen_width),
            action_size=action_size,
            shared_layers=[512, 256, 128, 64],
            critic_hidden_layers=[256, 128],
            actor_hidden_layers=[256, 128],
            init_type='xavier-uniform',
            seed=0
        ).to(device)
    
    # load the model
    checkpoint = torch.load(PATH)
    policy.load_state_dict(checkpoint['policy_state_dict'])

    # evaluate the model
    print("\nTest episodes:")
    rewards_list = []
    for e in range(episodes):
        rewards = eval_policy(envs=env, action_size=action_size, policy=policy)
        reward = np.sum(rewards,0)
        rewards_list.append(reward)
        #print("Episode: {0:d}, reward: {1}".format(e+1, reward), end="\n")
        print(f"Episode: {e+1}, reward: {reward}")
        
    rewards_true = rewards_list.count(1)
    percent_true = (rewards_true / episodes) * 100
    rewards_false = rewards_list.count(0)
    percent_false = (rewards_false / episodes) * 100
    
    print("\nThe final result of the training:")
    print(f"Count of all test episodes: {episodes}")
    print(f"True  episode count:   {rewards_true}")
    print(f"True  episode percent: {percent_true}%")
    print(f"False episode count:   {rewards_false}")
    print(f"False episode percent: {percent_false}%\n")
    
    