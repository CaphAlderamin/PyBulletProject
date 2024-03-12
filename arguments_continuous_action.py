import argparse
import multiprocessing as mp
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()
    # Algorithm arguments
    parser.add_argument("--anneal-lr", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, 
                        help="if toggled, learning rate annealing for policy and value networks.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, 
                        help="the learning rate of the optimazer.")
    
    parser.add_argument("--gae", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, 
                        help="if toggled, GAE is been used for advantage computation.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, 
                        help="tha lambda for the general advantage estimation.")
    
    parser.add_argument("--discount-gamma", type=float, default=0.99, 
                        help="the discount factor gamma.")
    parser.add_argument("--ratio-epsilon", type=float, default=0.07, 
                        help="the ratio used to clip r=new_probs/old_probs during training.")
    parser.add_argument("--entropy-beta", type=float, default=0.01, 
                        help="the entropy coefficient beta")
    
    parser.add_argument("--num-minibatches", type=int, default=32, 
                        help="the number of mini-batches.")
    parser.add_argument("--update-epochs", type=int, default=10, 
                        help="the K epochs to update the policy.")
    
    parser.add_argument("--norm-adv", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, 
                        help="if toggled, advantages normalization will be used.")
    parser.add_argument("--clip-coef", type=float, default=0.2, 
                        help="the surrogate clipping coefficient (ppo clipped policy objective).")
    parser.add_argument("--clip-vloss", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, 
                        help="if toggled, clipped loss will be used for the value function, as per the paper.")
    
    parser.add_argument("--ent-coef", type=float, default=0.0, 
                        help="coefficient of the entropy.")
    
    parser.add_argument("--vf-coef", type=float, default=0.5, 
                        help="coefficient of the value function.")
    
    parser.add_argument("--max-grad-norm", type=float, default=0.5, 
                        help="the maximum norm for the gradient clipping.")
    
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold.")
    parser.add_argument("--early-stopping", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, 
                        help="if toggled, the learning of the model will be stoped if it does not improve.")
    parser.add_argument("--early-stopping-season-start", type=int, default=25,
                        help="starting step count before early stopping is activated.")
    parser.add_argument("--early-stopping-mean-reward", type=int, default=50,
                        help="number of not improving model updates before early stopping.")
    #parser.add_argument("--early-stopping-min-delta", type=int, default=1e-6,
    #                    help="occuracy of improvement needed in order to keep training or execute early stopping.")
    
    parser.add_argument("--resize-resolution", type=int, default=48,
                        help="resize image resolution while preprocessing input images.")
    parser.add_argument("--scores-window", type=int, default=100, #default=100
                        help="window for mean model reward.")
    # Model arguments
    parser.add_argument("--model-save", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, torch model will be saved (check out 'models' folder).")
    # Environment arguments
    parser.add_argument("--num-envs", type=int, default=20, #default=mp.cpu_count()
                        help="the number of parallel game environment.")
    parser.add_argument("--num-steps", type=int, default=1024, #default=1024
                        help="the number of steps to run in each environment per policy rollout.")
    parser.add_argument("--exp-name", type=str, default="ppo_continuous_action", 
                        help="the name of this experiment.")
    parser.add_argument("--gym-id", type=str, default="KukaDiverseObjectEnv", #default="KukaDiverseObjectGrasping-v0"
                        help="the id of the gym environment.")
    parser.add_argument("--seed", type=int, default=1, 
                        help="seed of the experiment.")
    #parser.add_argument("--total-timesteps", type=int, default=2000000, 
    #                    help="total timesteps of the experiments.")
    parser.add_argument("--total-seasons", type=int, default=1000000, #default=1000000
                        help="total timesteps of the experiments.")
    # GPU arguments
    parser.add_argument("--torch-deterministic", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, 
                        help="if toggled, torch.backends.cudnn.deterministic=False.")
    parser.add_argument("--cuda", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, 
                        help="if toggled, cuda will not be enabled by default.")
    # Logging
    parser.add_argument("--info", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, additional information will be printed.")
    parser.add_argument("--info-camera", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, additional camera demonstration will be shown.")
    parser.add_argument("--track", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True, 
                        help="if toggled, this experiment will be tracked with clearml.")
    parser.add_argument("--clearml-project-name", type=str, default="FQW", 
                        help="the clearml's project name.")
    parser.add_argument("--clearml-task-name", type=str, default=None, 
                        help="the clearml's task name.")
    parser.add_argument("--clearml-tags", type=str, nargs='+', default=['PyTorch','PPO','Pybullet','KukaDiverseObjectEnv'], 
                        help="the clearml's task tags.")
    parser.add_argument("--capture-video", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True, 
                        help="wether to capture videos of the agent performances (check out 'videos' folder).")
    
    args = parser.parse_args()
    
    #args.batch_size = int(args.num_envs * (args.num_steps / args.num_envs))
    #args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
    args.batch_size = int(args.num_steps // args.num_minibatches)
    
    return args 


    #parser.add_argument("--", type=, default=, 
    #                    help="")