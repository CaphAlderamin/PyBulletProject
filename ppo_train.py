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

# cat all agents
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
        print("\nClearMl info:")
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
    
    # create directry for models
    if args.model_save:
        os.makedirs(f"models/{run_name}", exist_ok=True)
    
    # set image resolution resize 
    resize = T.Compose([
        T.ToPILImage(),
        T.Resize(args.resize_resolution, interpolation=Image.BICUBIC),
        T.ToTensor()
    ])
    # set the calculation device (cuda:0/cpu)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # create parallel PyBullet envs
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
    # reset all parallel PyBullet envs
    envs.reset()

    # number of agents
    num_agents = envs.num_envs
    # number of image channels
    num_channels = envs.observation_space.shape[-1]
    # size of each action
    action_size = envs.action_space.shape[0]
    # size of screen
    init_screen = envs.get_screen().to(device)
    _, _, screen_height, screen_width = init_screen.shape
    
    # set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # robot arm camera demonstration
    if args.info_camera:
        plt.figure()
        plt.imshow(init_screen[0].cpu().squeeze(0).permute(1, 2, 0).numpy(),
                interpolation='none')
        plt.title('Example extracted screen')
        plt.show()
    
    # run policy!
    #policy=ActorCritic(
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
        critic_hidden_layers=[512],
        actor_hidden_layers=[512],
        init_type='xavier-uniform',
        seed=args.seed
    ).to(device)

    # the adam optimizer with learning rate 3e-4 (defoult) (optim.SGD is also possible)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    
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
        print(policy, "\n")
        print(optimizer, "\n")
    
    # set up the models non constant parameters
    #discount = args.discount_gamma
    epsilon = args.ratio_epsilon
    beta = args.entropy_beta
    #opt_epoch = args.update_epochs
    #season = 10 #season = args.total_seasons
    #batch_size = args.batch_size #batch_size = 128
    #tmax = args.num_steps//num_agents #env episode steps
    #tmax = 40//num_agents
    
    # start the timer
    start_time = timeit.default_timer()
    
    best_mean_reward = None
    scores_window = deque(maxlen=args.scores_window)  # last "100" scores
    save_scores = []

    # model learning
    print("\nModel learning start:")
    for s in range(args.total_seasons):
        # annealing the learning rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (s - 1.0) / args.total_seasons
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        policy.eval()
        
        old_probs_lst, states_lst, actions_lst, rewards_lst,\
        values_lst, dones_list = collect_trajectories(
            envs=envs,
            policy=policy, 
            num_agents=num_agents, 
            action_size=action_size,
            tmax=args.num_steps//num_agents,
            nrand = 5,
            seed=args.seed, 
            device = device
        )
        
        season_score = rewards_lst.sum(dim=0).sum().item()
        scores_window.append(season_score)
        save_scores.append(season_score)
        
        gae, target_value = calc_returns(
            rewards = rewards_lst,
            values = values_lst,
            dones=dones_list,
            device = device
        )
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

        policy.train()

        old_probs_lst = concat_all(old_probs_lst)
        states_lst = concat_all(states_lst)
        actions_lst = concat_all(actions_lst)
        rewards_lst = concat_all(rewards_lst)
        values_lst = concat_all(values_lst)
        gae = concat_all(gae)
        target_value = concat_all(target_value)
        
        # gradient ascent step
        n_sample = len(old_probs_lst)//args.batch_size
        idx = np.arange(len(old_probs_lst))
        np.random.shuffle(idx)
        for epoch in range(args.update_epochs):
            for b in range(n_sample):
                ind = idx[b*args.batch_size:(b+1)*args.batch_size]
                g = gae[ind]
                tv = target_value[ind]
                actions = actions_lst[ind]
                old_probs = old_probs_lst[ind]

                action_est, values = policy(states_lst[ind])
                sigma = nn.Parameter(torch.zeros(action_size))
                dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(device))
                log_probs = dist.log_prob(actions)
                log_probs = torch.sum(log_probs, dim=-1)
                entropy = torch.sum(dist.entropy(), dim=-1)

                ratio = torch.exp(log_probs - old_probs)
                ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
                L_CLIP = torch.mean(torch.min(ratio*g, ratio_clipped*g))
                # entropy bonus
                S = entropy.mean()
                # squared-error value function loss
                L_VF = 0.5 * (tv - values).pow(2).mean()
                # clipped surrogate
                L = -(L_CLIP - L_VF + beta*S)
                
                optimizer.zero_grad()
                # This may need retain_graph=True on the backward pass
                # as pytorch automatically frees the computational graph after
                # the backward pass to save memory
                # Without this, the chain of derivative may get lost
                L.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step()
                del(L)

        # the clipping parameter reduces as time goes on
        epsilon*=.999
        # the regulation term also reduces
        # this reduces exploration in later runs
        beta*=.998

        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], s)
        writer.add_scalar("epsilon", epsilon, s)
        writer.add_scalar("beta", beta, s)
        mean_reward = np.mean(scores_window)
        writer.add_scalar("Season score", season_score, s)
        writer.add_scalar("100 episode mean score", mean_reward, s)
        
        # display some progress every n iterations
        elapsed = timeit.default_timer() - start_time
        print(f"Season {s} end, with score {int(season_score)}, Score: {mean_reward:.3f}, Elapsed time: {timedelta(seconds=elapsed)}")
        
        if best_mean_reward is None or best_mean_reward+1 < mean_reward:
                    # For saving the model and possibly resuming training
                    if args.model_save:
                        torch.save({
                                'policy_state_dict': policy.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
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
        plt.ylabel('Score')
        plt.xlabel('Season #')
        plt.grid()
        plt.show()
    
    writer.close()
    envs.close()
    if args.track:
        task.close()