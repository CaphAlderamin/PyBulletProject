import matplotlib.pyplot as plt

import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

from collections import deque
from datetime import timedelta
import random
import time
import timeit
from clearml import Task

from parallel_envs_creation import *
from actor_critic_model import *
from model_run import *

from arguments_continuous_action import parse_args

# Cat all agents
def concat_all(v):
    #print(v.shape)
    if len(v.shape) == 3:#actions
        return v.reshape([-1, v.shape[-1]])
    if len(v.shape) == 5:#states
        v = v.reshape([-1, v.shape[-3], v.shape[-2],v.shape[-1]])
        #print(v.shape)
        return v
    return v.reshape([-1])

if __name__ == "__main__":
    args = parse_args()
    if args.info:
        print(f"\nArguments info: \n  {args}") 
        
    # Set logging experiment name
    if args.clearml_task_name != None:
        run_name = args.clearml_task_name
    else:
        #run_name = f"{args.gym_id}_{args.exp_name}_s{args.seed}_t{int(time.time())}"
        run_time = time.strftime("%m-%d-%Y_%H-%M-%S", time.localtime())
        run_name = f"{args.gym_id}_{args.exp_name}_s{args.seed}_{run_time}"
        
    # Create directry for models
    if args.model_save:
        os.makedirs(f"models/{run_name}", exist_ok=True)    
        
    # Init ClearML logging
    if args.track:
        from clearml import Task
        print("\nClearMl info:")
        task = Task.init(
            project_name=args.clearml_project_name, 
            task_name=f'{run_name}', 
            tags=args.clearml_tags
        )
        task.connect(args)
    # Init TensorBoard logging
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
   
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # Set the calculation device (cuda:0/cpu)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Set image resolution resize 
    resize = T.Compose([
        T.ToPILImage(),
        T.Resize(args.resize_resolution, interpolation=Image.BICUBIC),
        T.ToTensor()
    ])
    
    # Create parallel PyBullet envs
    print("\nMultiprocessing info:")
    envs = make_batch_env(
        test=False,            
        resize=resize, 
        num_envs=args.num_envs,
        capture_video=args.capture_video,
        seed=args.seed,
        run_name=run_name,
        device=device
    )
    # Reset all parallel PyBullet envs
    start_obs = envs.reset()
    #obs = torch.Tensor(envs.reset()[0][0]).to(device)

    # Number of agents
    num_agents = envs.num_envs
    # Number of image channels
    #num_channels = envs.observation_space.shape[-1]
    num_channels = start_obs.shape[1]
    # Size of each action
    action_size = envs.action_space.shape[0]
    # Size of screen
    #init_screen = envs.get_screen().to(device)
    #_, _, screen_height, screen_width = init_screen.shape
    screen_height = start_obs.shape[2]
    screen_width = start_obs.shape[3]
    
    # Robot arm camera demonstration
    #if args.info_camera:
    #    plt.figure()
    #    plt.imshow(obs.cpu().squeeze(0).permute(1, 2, 0).numpy(),
    #            interpolation='none')
    #    plt.title('Example extracted screen')
    #    plt.show()
        
    # Create policy!
    #policy=ActorCritic(
    #    channels=num_channels,
    #    state_size=(screen_height, screen_width),
    #    action_size=action_size,
    #    shared_layers=[128, 64],
    #    critic_hidden_layers=[64],
    #    actor_hidden_layers=[64],
    #    init_type='xavier-uniform',
    #    seed=0
    #).to(device)
    #agent=ActorCritic(
    #    channels=num_channels,
    #    state_size=(screen_height, screen_width),
    #    action_size=action_size,
    #    shared_layers=[512, 256, 128, 64],
    #    critic_hidden_layers=[512],
    #    actor_hidden_layers=[512],
    #    init_type='xavier-uniform',
    #    seed=args.seed
    #).to(device)
    agent=ActorCritic(
        envs = envs,
        observation_size = (screen_height, screen_width),
        #init_type='xavier-uniform',
    ).to(device)
    # The adam optimizer with learning rate 3e-4 (default) (optim.SGD is also possible)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    #print(f"screen_height {screen_height}")
    #print(f"screen_width {screen_width}")
    
    #print(f"init_screen {init_screen}")
    #print(f"obs { obs.shape}")
    #screen = obs.transpose((2, 0, 1))
    #screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    #screen = torch.from_numpy(screen)
    #screen = resize(screen).unsqueeze(0)
    #print(f"screen {screen}")
    #print(f"screen {screen.shape}")
    observation_space_shape = start_obs.shape
    
    # variables for saving model
    #policy_shared_layers = [layer.out_features for layer in agent.shared_layers]
    #policy_critic_hidden_layers = [layer.out_features for layer in agent.critic_hidden]
    #policy_actor_hidden_layers = [layer.out_features for layer in agent.actor_hidden]
    
    # print additional info
    if args.info:
        print("\nGPU info:")
        print(f"  selected device: {device}")
        
        print("\nAgents info:")
        print(f'  number_of_agents/envs: {num_agents}/{args.num_envs}')
    
        print("\nEnvs info:")
        print(f"  action_space: {envs.action_space}")
        print(f"  each_action_size: {action_size}")
        print(f"\n  observation_space: {envs.observation_space}")
        print(f"  num_channels: {num_channels}")
        print(f"  screen_height: {screen_height}")
        print(f"  screen_width:  {screen_width}")
        
        print("\nAgent info:")
        #print(f"policy.shared_layers: {policy_shared_layers}")
        #print(f"policy.critic_hidden_layers: {policy_critic_hidden_layers}")
        #print(f"policy.actor_hidden_layers: {policy_critic_hidden_layers}")
        #print(f"policy.init_type: {agent.init_type}\n")
        print(agent, "\n")
        print(optimizer, "\n")
    
    # Start the timer
    #start_time = timeit.default_timer()
    
    best_mean_reward = None
    best_mean_reward_percent = None
    scores_window = deque(maxlen=args.scores_window)  # last "100" scores
    scores_window_percent = deque(maxlen=args.scores_window)  # last "100" scores
    save_scores = []
    save_scores_percent = []
    update_len = []

    # storage setup
    #obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    #obs = torch.zeros((args.num_steps, args.num_envs) + obs.shape).to(device)
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Set up the models parameters
    global_step = 0
    start_time = time.time()
    #next_obs = torch.Tensor(envs.reset()).to(device)
    next_obs = torch.Tensor(start_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    #epsilon = 0.07
    #beta = 0.01
    #tmax = args.num_steps//num_agents #env episode steps
    

    # model learning
    print("\nModel learning start:")
    for update in range(1, num_updates + 1):
        update_scores = []
        
        # annealing the learning rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            
        for step in range(0, args.num_steps):
            global_step += 1 * num_agents
            obs[step] = next_obs
            dones[step] = next_done
    
            # action logic
            with torch.no_grad():
                #print(f"next_obs ({update}): {next_obs}")
                #print(f"next_obs ({update}): {next_obs.shape}")
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            #print(f"action.shape {action.shape}")
            #print(f"actions.shape {actions.shape}")
            #print(f"envs.action_space: {envs.action_space}")
            #print(f"envs.action_space.shape {envs.action_space.shape}")
            actions[step] = action
            logprobs[step] = logprob
            
            # execute the game step and log data
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            #next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            # TODO : "next_obs[0]" временная мера для для дебага, на самомо деле в 
            # next_obs соедржится инфорация о n=cpu.threads количествах выполнения step()
            # пока что там только 1 поток.
            #print(f"next_obs {next_obs[0]}")
            #print(f"next_obs {next_obs[0].shape}")
            #next_obs = torch.tensor(next_obs[0]).to(device)
            next_obs = next_obs.clone().detach().to(device)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = torch.Tensor(done).to(device)
            #next_terminated = torch.tensor(terminated).to(device)
            #next_truncated = torch.tensor(truncated).to(device)
            
            #print(step)
            #print(info)
            #print(next_done)
            #print(next_done.any().item())
            
            
            #if info != {}:        
            if next_done.any().item():
            #if next_done.all().item():
            #if step % 8 == 0 and step != 0:
                episodic_return = sum(d.get('episode', {}).get('r', 0) for d in info)
                episodic_length = next((d['episode'].get('l') for d in info if 'episode' in d), None)
                
                update_scores.append(episodic_return)
                print(f"global_step: {global_step}, " +
                      f"rewards: {int(episodic_return): <2}/{len(next_done)}, " +
                      f"length {int(episodic_length)}")
                
                #episodic_return_norm = episodic_return / num_agents
                #update_scores.append(episodic_return_norm)
                #print(f"global_step: {global_step}, episodic_return: {episodic_return_norm} ({int(episodic_return)}/{num_agents})")
                
                writer.add_scalar("episodic_return", episodic_return, global_step)
                writer.add_scalar("episodic_length", episodic_length, global_step)
                
                # TODO : узнать что возвращается в step() если заканчивается эпизод в SyncVectorEnv
                next_obs = envs.reset().clone().detach().to(device)
                # TODO : this line need for debug
                #break 
             
            #if next_terminated.any().item() or next_truncated.any().item():
            #    for item in info['final_info']:
            #        if item != None:
            #            episodic_return = item['episode']['r']
            #            update_episodic_return_total += episodic_return     
            #            update_episodic_count += 1
            #            print(f"global_step: {global_step}, episodic_return: {episodic_return}")
            #            writer.add_scalar("episodic_return", episodic_return, global_step)
            #            writer.add_scalar("episodic_length", episodic_return, global_step)
            #            break
        
        # save update(season) scores
        
        update_score = sum(update_scores)
        update_score_max = (len(update_scores) * num_agents)
        update_score_percent = update_score / update_score_max * 100
        scores_window.append(update_score)
        scores_window_percent.append(update_score_percent)
        save_scores.append(update_score)
        save_scores_percent.append(update_score_percent)
        update_len.append(len(update_scores) * num_agents)
        
        #if update >= 2:
        #    print(f"update_scores {update_scores}")
        #    print(f"update_score {update_score}")
        #    print(f"scores_window {scores_window}")
        #    print(f"save_scores_percent {save_scores_percent}")
        #    print(f"save_scores {save_scores}")
        #    print(f"save_scores_percent {save_scores_percent}")
        #    print(f"update_scores_len {update_len}")
        #    raise
            
        # bootstrap reward if not terminated (original ppo repo code)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        #nextnonterminal = 1.0 - next_terminated.any().item()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        #nextnonterminal = 1.0 - next_terminated.any().item()
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
    
        # flatten the batch
        #b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        #print(f"observation_space_shape {tuple(observation_space_shape)[1:]}")
        b_obs = obs.reshape((-1,) + tuple(observation_space_shape)[1:])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
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
                #loss.backward(retain_graph=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # TODO early stopping realization not working well
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    print(f"----approx_kl > args.target_kl----: {approx_kl}")
                    break
            
        # another early stopping realization
        #if args.early_stopping:
        #    if global_step > args.early_stopping_step_start:
        #        early_stopping(v_loss)
        #        if args.info:
        #            print(f"current_value_loss: {v_loss.item():.9f}")
        #            print(f"minimum_value_loss: {early_stopping.best_score.item():.9f}, count: ({early_stopping.counter})")
        #        if args.model_save and early_stopping.counter == 0:
        #            best_agent_state_dict = agent.state_dict()
        #            best_optimizer_state_dict = optimizer.state_dict()
        #            if args.info:
        #                print("SAVING_THE_CURRENT_MODEL_STATE_DICT")
        #        if early_stopping.early_stop:
        #            print("Early stopping")
        #            break
            
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # logging rewards for ploting        
        mean_reward = np.mean(scores_window)
        writer.add_scalar("mean_score", mean_reward, global_step)
        mean_reward_percent = np.mean(scores_window_percent)
        writer.add_scalar("mean_score_percent", mean_reward_percent, global_step)
        writer.add_scalar("update_score", update_score, global_step)
        writer.add_scalar("update_score_percent", update_score_percent, global_step)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("value_loss", v_loss.item(), global_step)
        writer.add_scalar("policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("entropy", entropy_loss.item(), global_step)
        writer.add_scalar("old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("SPS", sps, global_step)
        
        print(f"Update {update} done, " +
              f"SPS: {sps}, " +
              f"Score {int(update_score)}/{int(update_score_max)} ({update_score_percent:.3f}%), " +
              f"Mean score: {mean_reward:.3f} ({mean_reward_percent:.3f}%), " +
              f"Elapsed time: {timedelta(seconds=time.time() - start_time)}")
        
        # saving the best model during training
        # TODO: saving the model during the training takes too much time (currently model save after all training updates)
        #if update_episodic_return_mean_new > update_episodic_return_mean_current:
        #    update_episodic_return_mean_current = update_episodic_return_mean_new
        #    print(f"new_maximum_average_episodic_return: {update_episodic_return_mean_current}")
        #if best_mean_reward is None or int(best_mean_reward)+1 < mean_reward:    
        if best_mean_reward_percent is None or int(best_mean_reward_percent)+1 < mean_reward_percent:    
            if args.model_save:
                torch.save({
                        # Base
                        'policy_state_dict': agent.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # Model reload variables
                        #'policy.shared_layers': policy_shared_layers,
                        #'policy.critic_hidden_layers': policy_critic_hidden_layers,
                        #'policy.actor_hidden_layers': policy_critic_hidden_layers,
                        #'policy.init_type': agent.init_type,
                        # Learning variables                    
                        'lr': optimizer.param_groups[0]["lr"],
                        #'epsilon': epsilon,
                        #'beta': beta
                    }, f"models/{run_name}/best_model.pt")
                #print("best_model saved")
            print(f"Best mean reward updated {(best_mean_reward if best_mean_reward is not None else 0.):.3f} -> {mean_reward:.3f} " +
                  f"({(best_mean_reward_percent if best_mean_reward_percent is not None else 0.):.3f}% -> {mean_reward_percent:.3f}%)"
                  f"{', best_model saved' if args.model_save else ''}")
                
            best_mean_reward = mean_reward
            best_mean_reward_percent = mean_reward_percent

        if args.early_stopping:
            if update>=args.early_stopping_season_start \
            and mean_reward_percent>args.early_stopping_mean_reward:
                print(f"Environment solved in {update} seasons!")
                break
        
    if args.model_save:
        torch.save({
                # Base
                'policy_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # Model reload variables
                #'policy.shared_layers': policy_shared_layers,
                #'policy.critic_hidden_layers': policy_critic_hidden_layers,
                #'policy.actor_hidden_layers': policy_critic_hidden_layers,
                #'policy.init_type': agent.init_type,
                # Learning variables                    
                'lr': optimizer.param_groups[0]["lr"],
                #'epsilon': epsilon,
                #'beta': beta
            }, f"models/{run_name}/complete_model.pt")
        print("complete_model saved")
        
    # close all environments
    envs.close()
    # close tensorboard logging
    writer.close()
    # close clearml logging
    if args.track:
        task.close()
    print()