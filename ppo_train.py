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

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

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
    
    # Create directry for models
    if args.model_save:
        os.makedirs(f"models/{run_name}", exist_ok=True)
    
    # Set image resolution resize 
    resize = T.Compose([
        T.ToPILImage(),
        T.Resize(args.resize_resolution, interpolation=Image.BICUBIC),
        T.ToTensor()
    ])
    # Set the calculation device (cuda:0/cpu)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
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
    envs.reset()

    # Number of agents
    num_agents = envs.num_envs
    # Number of image channels
    num_channels = envs.observation_space.shape[-1]
    # Size of each action
    action_size = envs.action_space.shape[0]
    # Size of screen
    init_screen = envs.get_screen().to(device)
    _, _, screen_height, screen_width = init_screen.shape
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # Robot arm camera demonstration
    if args.info_camera:
        plt.figure()
        plt.imshow(init_screen[0].cpu().squeeze(0).permute(1, 2, 0).numpy(),
                interpolation='none')
        plt.title('Example extracted screen')
        plt.show()
    
    # run policy!
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
    policy=ActorCritic(
        channels=num_channels,
        state_size=(screen_height, screen_width),
        action_size=action_size,
        shared_layers=[512, 256, 128, 64],
        #shared_layers=[512, 256, 128],
        critic_hidden_layers=[512],
        #critic_hidden_layers=[64],
        actor_hidden_layers=[512],
        #actor_hidden_layers=[64],
        init_type='xavier-uniform',
        seed=args.seed
    ).to(device)
    # The adam optimizer with learning rate 3e-4 (defoult) (optim.SGD is also possible)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    
    policy.conv1.register_forward_hook(get_activation('conv1'))
    policy.conv2.register_forward_hook(get_activation('conv2'))
    policy.conv3.register_forward_hook(get_activation('conv3'))    
    
    # variables for saving model
    policy_shared_layers = [layer.out_features for layer in policy.shared_layers if isinstance(layer, nn.Linear)]
    policy_critic_hidden_layers = [layer.out_features for layer in policy.critic_hidden if isinstance(layer, nn.Linear)]
    policy_actor_hidden_layers = [layer.out_features for layer in policy.actor_hidden if isinstance(layer, nn.Linear)]
    
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
        print(f"policy.shared_layers: {policy_shared_layers}")
        print(f"policy.critic_hidden_layers: {policy_critic_hidden_layers}")
        print(f"policy.actor_hidden_layers: {policy_critic_hidden_layers}")
        print(f"policy.init_type: {policy.init_type}\n")
        print(policy, "\n")
        print(f'Policy parameters: {count_parameters(policy)}\n')
        print(optimizer, "\n")
    
    
    # Set up the models non constant parameters
    #discount = args.discount_gamma
    epsilon = args.ratio_epsilon
    beta = args.entropy_beta
    #opt_epoch = args.update_epochs
    #season = 10 #season = args.total_seasons
    #batch_size = args.batch_size #batch_size = 128
    tmax = args.num_steps//num_agents #env episode steps
    #tmax = 40//num_agents
    
    # Start the timer
    start_time = timeit.default_timer()
    
    best_mean_reward = None
    best_mean_reward_percent = None
    scores_window = deque(maxlen=args.scores_window)  # last "100" scores
    scores_window_percent = deque(maxlen=args.scores_window)
    save_scores = []
    save_scores_percent = []
    season_lengths = []

    # model learning
    print("\nModel learning start:")
    for s in range(args.total_seasons):
        # Annealing the learning rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (s - 1.0) / args.total_seasons
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            # the clipping parameter reduces as time goes on
            epsilon*=.999
            # the regulation term also reduces
            # this reduces exploration in later runs
            beta*=.998
        
        policy.eval()
        
        old_probs_lst, states_lst, actions_lst, rewards_lst,\
        rewards_end_lst, values_lst, dones_list = collect_trajectories(
            envs=envs,
            policy=policy, 
            num_agents=num_agents, 
            action_size=action_size,
            tmax=tmax,
            nrand = 5,
            seed=args.seed, 
            device = device
        )
        
        #num_images = len(states_lst)
        #fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        #for ax, states in zip(axes, states_lst):
        #    ax.imshow(states[0].cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
        #    ax.set_title('Example extracted screen')
        #    ax.axis('off')
        #plt.show()
        
        #output_dir = 'output_images'
        #os.makedirs(output_dir, exist_ok=True)
        #for idx, states in enumerate(states_lst, 1):
        #    plt.figure()
        #    plt.imshow(states[0].cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
        #    plt.title('Example extracted screen')
        #    plt.axis('off')
        #    plt.savefig(os.path.join(output_dir, f'{idx}.png'))
        #    plt.close()
        
        season_length = rewards_end_lst.numel()
        season_lengths.append(season_length)
        
        season_score = rewards_end_lst.sum(dim=0).sum().item()
        scores_window.append(season_score)
        save_scores.append(season_score)
        
        season_score_percent = (season_score / season_length) * 100
        scores_window_percent.append(season_score_percent)
        save_scores_percent.append(season_score_percent)
        
        #print(season_score)
        #print(season_length)
        #print(season_accuracy)
        
        gae, target_value = calc_returns(
            rewards = rewards_lst,
            values = values_lst,
            dones=dones_list,
            gae_lambda=args.gae_lambda,
            discount_gamma=args.discount_gamma,
            device = device
        )
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

        policy.train()

        # Concatenate tensors
        old_probs_lst = concat_all(old_probs_lst)
        states_lst = concat_all(states_lst)
        actions_lst = concat_all(actions_lst)
        rewards_lst = concat_all(rewards_lst)
        rewards_end_lst = concat_all(rewards_end_lst)
        values_lst = concat_all(values_lst)
        gae = concat_all(gae)
        target_value = concat_all(target_value)
        
        # Gradient ascent step
        n_sample = len(old_probs_lst)//args.batch_size
        idx = np.arange(len(old_probs_lst))
        
        for epoch in range(args.update_epochs):
            np.random.shuffle(idx)
            for b in range(n_sample):
                ind = idx[b*args.batch_size:(b+1)*args.batch_size]
                g = gae[ind]
                tv = target_value[ind]
                actions = actions_lst[ind]
                old_probs = old_probs_lst[ind]

                # Policy network
                action_est, values = policy(states_lst[ind])
                
                sigma = nn.Parameter(torch.zeros(action_size))
                sigma = F.softplus(sigma).to(device)
                dist = torch.distributions.Normal(action_est, sigma)
                
                log_probs = dist.log_prob(actions)
                log_probs = torch.sum(log_probs, dim=-1)
                
                entropy = torch.sum(dist.entropy(), dim=-1)

                # PPO loss calculation
                ratio = torch.exp(log_probs - old_probs)
                ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
                L_CLIP = torch.mean(torch.min(ratio*g, ratio_clipped*g))
                # entropy bonus
                S = entropy.mean()
                # squared-error value function loss
                L_VF = 0.5 * (tv - values).pow(2).mean()
                
                # clipped surrogate
                loss = -(L_CLIP - L_VF + beta * S)
                
                optimizer.zero_grad()
                # This may need retain_graph=True on the backward pass
                # as pytorch automatically frees the computational graph after
                # the backward pass to save memory
                # Without this, the chain of derivative may get lost
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step()
                
                # Сохранение активаций в папку output_conv
                #if b > 5: raise
                #for layer_name, activation in activations.items():
                #    #print(f'Layer: {layer_name}, Shape: {activation.shape}')
                #    img = activation[0, 0].cpu().numpy()
                #    plt.imshow(img, cmap='gray')
                #    #plt.title(layer_name)
                #    output_path = os.path.join('output_conv_16', f'({epoch}-{b})_{layer_name}_{activation.shape}.png')
                #    plt.axis('off')
                #    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                #    plt.close()
                

        y_pred = values_lst.cpu().numpy()
        y_true = target_value.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], s)
        writer.add_scalar("epsilon", epsilon, s)
        writer.add_scalar("beta", beta, s)
        mean_reward = np.mean(scores_window)
        writer.add_scalar("Score", mean_reward, s)
        mean_reward_percent = np.mean(scores_window_percent)
        writer.add_scalar("score percent", mean_reward_percent, s)
        writer.add_scalar("season score", season_score, s)
        writer.add_scalar("season score percent", season_score_percent, s)
        writer.add_scalar("explained_variance", explained_var, s)

        # display some progress every n iterations
        elapsed = timeit.default_timer() - start_time
        print(f"Season {s} end, " +
              f"with score {int(season_score)}/{int(season_length)} ({season_score_percent:.3f}%), " +
              f"Score: {mean_reward:.3f} ({mean_reward_percent:.3f}%), " +
              f"Elapsed time: {timedelta(seconds=elapsed)}")
        
        if best_mean_reward is None or best_mean_reward+1 < mean_reward:
                    # For saving the model and possibly resuming training
                    if args.model_save:
                        torch.save({
                                # Base
                                'policy_state_dict': policy.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                # Model reload variables
                                'policy.shared_layers': policy_shared_layers,
                                'policy.critic_hidden_layers': policy_critic_hidden_layers,
                                'policy.actor_hidden_layers': policy_critic_hidden_layers,
                                'policy.init_type': policy.init_type,
                                # Learning variables                    
                                'lr': optimizer.param_groups[0]["lr"],
                                'epsilon': epsilon,
                                'beta': beta
                                }, f"models/{run_name}/complete_model.pt")
                    if best_mean_reward is not None:
                        print(f"Best mean reward updated {best_mean_reward:.3f} -> {mean_reward:.3f}"
                              f"{', model saved' if args.model_save else ''}")
                    best_mean_reward = mean_reward
        
        if args.early_stopping:
            if s>=args.early_stopping_season_start \
            and mean_reward>args.early_stopping_mean_reward:
                print(f"Environment solved in {s+1:d} seasons!")
                break
        
    print(f"Average Score of training: {mean_reward:.3f}")
    elapsed = timeit.default_timer() - start_time
    print(f"Elapsed time of training: {timedelta(seconds=elapsed)}")
    
    # Plot score
    if args.info:
        fig = plt.figure()
        plt.plot(np.arange(len(save_scores)), save_scores)
        plt.ylabel('score')
        plt.xlabel('season #')
        plt.grid()
        plt.show()
    
    writer.close()
    envs.close()
    if args.track:
        task.close()