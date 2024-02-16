import matplotlib.pyplot as plt
import torch.optim as optim
from collections import deque
import timeit
from datetime import timedelta
from clearml import Task

from ParallelEnvsCreation import *
from ActorCriticModel import *
from ModelRun import *
from Attendants import select_device

camera_demonstration = False

if __name__ == "__main__":
    # Init ClearML auto logging
    task = Task.init(
        project_name='FQW', 
        task_name='PPO1', 
        tags=['PyTorch','Pybullet','KukaDiverseObjectEnv'])
    
    # Select a device for PyTorch calculations
    device = select_device(info=True)
    
    # Create parallel PyBullet envs
    envs = make_batch_env(test=False)
    # Reset all parallel PyBullet envs
    envs.reset()

    # Number of agents
    num_agents = envs.num_envs
    print('Number of agents:', num_agents)
    # Size of each action
    action_size = envs.action_space.shape[0]
    print('Size of each action:', action_size)
    # Size of screen
    init_screen = envs.get_screen().to(device)
    _, _, screen_height, screen_width = init_screen.shape
    
    # Robot arm camera demonstration
    if camera_demonstration:
        plt.figure()
        plt.imshow(init_screen[0].cpu().squeeze(0).permute(1, 2, 0).numpy(),
                interpolation='none')
        plt.title('Example extracted screen')
        plt.show()
    
    # run your own policy!
    #policy=ActorCritic(
    #            state_size=(screen_height, screen_width),
    #            action_size=action_size,
    #            shared_layers=[128, 64],
    #            critic_hidden_layers=[64],
    #            actor_hidden_layers=[64],
    #            init_type='xavier-uniform',
    #            seed=0
    #        ).to(device)
    
    policy=ActorCritic(
                state_size=(screen_height, screen_width),
                action_size=action_size,
                shared_layers=[512, 256, 128, 64],
                critic_hidden_layers=[256, 128],
                actor_hidden_layers=[256, 128],
                init_type='xavier-uniform',
                seed=0
            ).to(device)

    # we use the adam optimizer with learning rate 2e-4
    # optim.SGD is also possible
    optimizer = optim.Adam(policy.parameters(), lr=2e-4)
    
    PATH = 'models/policy_ppo_m50_big1.pt'
    
    # Learn Model
    print("\n\nModel learning start:")
    best_mean_reward = None

    scores_window = deque(maxlen=100)  # last 100 scores

    discount = 0.993
    epsilon = 0.07
    beta = .01
    opt_epoch = 10
    season = 1000000
    batch_size = 128
    tmax = 1000//num_agents#env episode steps
    save_scores = []
    start_time = timeit.default_timer()

    for s in range(season):
        policy.eval()
        old_probs_lst, states_lst, actions_lst, rewards_lst, values_lst, dones_list = collect_trajectories(envs=envs,
                                                                                                        policy=policy, 
                                                                                                        num_agents=num_agents, 
                                                                                                        action_size=action_size,
                                                                                                        tmax=tmax,
                                                                                                        nrand = 5)

        season_score = rewards_lst.sum(dim=0).sum().item()
        scores_window.append(season_score)
        save_scores.append(season_score)
        
        gea, target_value = calc_returns(rewards = rewards_lst,
                                        values = values_lst,
                                        dones=dones_list)
        gea = (gea - gea.mean()) / (gea.std() + 1e-8)

        policy.train()

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

        old_probs_lst = concat_all(old_probs_lst)
        states_lst = concat_all(states_lst)
        actions_lst = concat_all(actions_lst)
        rewards_lst = concat_all(rewards_lst)
        values_lst = concat_all(values_lst)
        gea = concat_all(gea)
        target_value = concat_all(target_value)
        
        # gradient ascent step
        n_sample = len(old_probs_lst)//batch_size
        idx = np.arange(len(old_probs_lst))
        np.random.shuffle(idx)
        for epoch in range(opt_epoch):
            for b in range(n_sample):
                ind = idx[b*batch_size:(b+1)*batch_size]
                g = gea[ind]
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

        mean_reward = np.mean(scores_window)
        writer.add_scalar("epsilon", epsilon, s)
        writer.add_scalar("beta", beta, s)
        writer.add_scalar("Score", mean_reward, s)
        # display some progress every n iterations
        if best_mean_reward is None or best_mean_reward < mean_reward:
                    # For saving the model and possibly resuming training
                    torch.save({
                            'policy_state_dict': policy.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epsilon': epsilon,
                            'beta': beta
                            }, PATH)
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward
        if s>=25 and mean_reward>50:
            print('Environment solved in {:d} seasons!\tAverage Score: {:.2f}'.format(s+1, mean_reward))
            break


    print('Average Score: {:.2f}'.format(mean_reward))
    elapsed = timeit.default_timer() - start_time
    print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
    writer.close()
    envs.close()
    
    # Plot score
    fig = plt.figure()
    plt.plot(np.arange(len(save_scores)), save_scores)
    plt.ylabel('Score')
    plt.xlabel('Season #')
    plt.grid()
    plt.show()
    
    task.close()