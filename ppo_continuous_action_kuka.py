import random
import time
import os

import gym
import pybullet_envs
from gym.utils import seeding
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from arguments_continuous_action import parse_args

def make_env(gym_id, seed, idx, capture_video, run_name):
    """Create a Gym environment with given parameters.

    Args:
        gym_id (str): The id of the Gym environment to create.
        seed (int): The initial seed for the environment's random number generator. 
            This is used for reproducibility.
        idx (int):  The index of this environment. When capturing video, only the environment
            with idx=0 will actually capture the video to avoid creating too many files.
        capture_video (bool): Whether or not to capture the video of the environment's gameplay.
        run_name (str): The name of the current run. This is used for generating the video file name.
        
    Returns:
    - A zero-argument function (thunk) that returns the Gym environment.
    """
    def thunk():
        #env = gym.make(gym_id, render = False)
        #env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.make(gym_id, removeHeightHack=False, maxSteps=40, render_mode="rgb_array", renders=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                #env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda t: t % 10 == 0)
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #print("====",env.observation_space.shape)
        #env = NoopResetEnv(env, noop_max=30)
        #env = MaxAndSkipEnv(env, skip=4)
        #env = EpisodicLifeEnv(env)
        #env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (48, 48)) # in KukaDiverseObjectGrasping-v0 Observation is already Box(0.0, 255.0, (48, 48, 3), float32)
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        #print("====",env.observation_space.shape)
        env.np_random = seeding.np_random(seed)[0]
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize a neural network layer with orthogonal weights and constant biases.

    Args:
        layer (torch.nn.Module): The layer to be initialized.
        std (float, optional): The standard deviation of the normal distribution 
            from which the weights are drawn. Defaults to the square root of 2.
        bias_const (float, optional): The constant value with which the biases 
            are initialized. Defaults to 0.0.

    Returns:
        torch.nn.Module: The same layer with initialized parameters.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 2 * 2, 512)),
            nn.ReLU(),
        )
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.actor_mean = layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01)
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
    
    def get_value(self, x):
        return self.critic(self.network(x / 255))

    def get_action_and_value(self, x, action=None):
        #print(f"==============x: {x}")
        #print(f"==============x: {x.shape}")
        hidden = self.network(x / 255)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        #print(f"action.shape {action.shape}") 
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden)

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = np.inf
        self.early_stop = False

    def __call__(self, score: float):
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        elif score >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

if __name__ == "__main__":
    # print used args
    args = parse_args()
    if args.info:
        print(f"\nArguments info: \n{args}")
    
    # set logging experiment name
    if args.clearml_task_name != None:
        run_name = args.clearml_task_name
    else:
        #run_name = f"{args.gym_id}_{args.exp_name}_s{args.seed}_t{int(time.time())}"
        run_time = time.strftime("%m-%d-%Y_%H-%M-%S", time.localtime())
        run_name = f"{args.gym_id}_{args.exp_name}_s{args.seed}_{run_time}"
        
    # init ClearML logging
    if args.track:
        from clearml import Task
        task = Task.init(
            project_name=args.clearml_project_name, 
            task_name=f'{run_name}', 
            tags=args.clearml_tags
        )
        task.connect(args)
    # init TensorBoard logging
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # set the calculation device (cuda:0/cpu)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.info:
        print("\nGPU info:")
        print(f"Selected device: {device}")
    
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    if args.info:
        print("\nEnv info:")
        print(f"number_of_envs : {len(envs.envs)}/{args.num_envs}")
        print(f"envs.single_observation_space.shape: {envs.single_observation_space.shape}")
        print(f"envs.single_action_space.n: {envs.single_action_space.shape}")
        
        print("\nAgent info:")
        print(agent)
    
    # storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    #dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    terminateds = torch.zeros((args.num_steps, args.num_envs)).to(device)
    truncateds = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    #next_done = torch.zeros(args.num_envs).to(device)
    next_terminated = torch.zeros(args.num_envs).to(device)
    next_truncated = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    if args.info:
        print("\nObservations info:")
        print(f"num_updates: {num_updates}")
        print(f"next_obs.shape: {next_obs.shape}")
        #print(f"agent.get_value(next_obs): \n{agent.get_value(next_obs)}")
        #print(f"agent.get_value(next_obs).shape: {agent.get_value(next_obs).shape}")
        #print(f"agent.network(next_obs) {agent.network(next_obs)}")
        #print(f"agent.network(next_obs).shape: {agent.network(next_obs).shape}")
        
        print("\nActions info:")
        #print(f"agent.get_action_and_value(next_obs): \n{agent.get_action_and_value(next_obs)}")
        print(f"next_obs {next_obs}")
        print(f"next_obs.shape {next_obs.shape}")
    
    # to know mean episodic_return in update and when model is better
    #update_episodic_return_mean_current = -np.inf
    # create directry for models
    if args.model_save:
        os.makedirs(f"models/{run_name}", exist_ok=True)
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.early_stopping_patience, min_delta=args.early_stopping_min_delta)
    
    #print(envs.single_observation_space.shape)
    #print(next_obs.shape)
    #print(agent.network(next_obs).shape)
    
    print("\nstart ppo loop:")
    for update in range(1, num_updates + 1):
        update_episodic_return_total  = 0
        update_episodic_count = 0
        
        # annealing the learning rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            #dones[step] = next_done
            terminateds[step] = next_terminated
            truncateds[step] = next_truncated
    
            # action logic
            with torch.no_grad():
                print(f"next_obs ({update}): {next_obs}")
                print(f"next_obs ({update}): {next_obs.shape}")
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            #print(f"action.shape {action.shape}")
            #print(f"actions.shape {actions.shape}")
            #print(f"envs.action_space: {envs.action_space}")
            #print(f"envs.action_space.shape {envs.action_space.shape}")
            actions[step] = action
            logprobs[step] = logprob
            
            # execute the game step and log data
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            #print(f"next_obs {next_obs[0].shape}")
            print(next_obs)
            
            next_obs = torch.tensor(next_obs).to(device)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_terminated = torch.tensor(terminated).to(device)
            next_truncated = torch.tensor(truncated).to(device)
            
            
            print(info)
            
            #if info != {}:        
            if next_terminated.any().item() or next_truncated.any().item():
                for item in info['final_info']:
                    if item != None:
                        episodic_return = item['episode']['r']
                        update_episodic_return_total += episodic_return     
                        update_episodic_count += 1
                        print(f"global_step: {global_step}, episodic_return: {episodic_return}")
                        writer.add_scalar("episodic_return", episodic_return, global_step)
                        writer.add_scalar("episodic_length", episodic_return, global_step)
                        break
                break
        
        # bootstrap reward if not terminated (original ppo repo code)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_terminated.any().item()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - terminateds[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_terminated.any().item()
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - terminateds[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
    
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                #print(f"b_obs[mb_inds] {b_obs[mb_inds]}")
                #print(f"b_obs[mb_inds] {b_obs[mb_inds].shape}")
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # calculate approx_kl (http://joschu.net/blog/kl-approx.html)
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # entropy loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # TODO early stopping realization not working well
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    print(f"----approx_kl > args.target_kl----: {approx_kl}")
                    break
            
        # another early stopping realization
        if args.early_stopping:
            if global_step > args.early_stopping_step_start:
                early_stopping(v_loss)
                if args.info:
                    print(f"current_value_loss: {v_loss.item():.9f}")
                    print(f"minimum_value_loss: {early_stopping.best_score.item():.9f}, count: ({early_stopping.counter})")
                if args.model_save and early_stopping.counter == 0:
                    best_agent_state_dict = agent.state_dict()
                    best_optimizer_state_dict = optimizer.state_dict()
                    if args.info:
                        print("SAVING_THE_CURRENT_MODEL_STATE_DICT")
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # saving the best model during training
        # TODO: saving the model during the training takes too much time (currently model save after all training updates)
        update_episodic_return_mean_new = update_episodic_return_total / update_episodic_count if update_episodic_count > 0 else 0       
        #if update_episodic_return_mean_new > update_episodic_return_mean_current:
        #    update_episodic_return_mean_current = update_episodic_return_mean_new
        #    print(f"new_maximum_average_episodic_return: {update_episodic_return_mean_current}")
        #    if args.model_save:
        #        torch.save({
        #            'policy_state_dict': agent.state_dict(),
        #            'optimizer_state_dict': optimizer.state_dict(),
        #            }, f"models/{run_name}/best_model.pt")

        # logging rewards for ploting        
        writer.add_scalar("update_return_mean", update_episodic_return_mean_new, global_step)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("value_loss", v_loss.item(), global_step)
        writer.add_scalar("policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("entropy", entropy_loss.item(), global_step)
        writer.add_scalar("old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        print(f"SPS: {sps}")
        writer.add_scalar("SPS", sps, global_step)
    
    if args.model_save:
        torch.save({
            'policy_state_dict': best_agent_state_dict,
            'optimizer_state_dict': best_optimizer_state_dict,
            }, f"models/{run_name}/best_model_vl{early_stopping.best_score.item()}.pt")
        print("best_model saved")
        torch.save({
            'policy_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f"models/{run_name}/complete_model_vl{v_loss.item()}.pt")
        print("complete_model saved")
        
    # close all environments
    envs.close()
    # close tensorboard logging
    writer.close()
    # close clearml logging
    if args.track:
        task.close()
    print()
    
    