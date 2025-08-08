
import numpy as np
import os

import functools
import multiprocessing as mp
from multiprocessing import Pipe
from multiprocessing import Process
import signal

import torch

import gym as gym
from gym import spaces
from gym.utils import seeding
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv  
import pybullet as p

def worker(remote, env_fn, resize, seed):
    # Ignore CTRL+C in the worker process
    #signal.signal(signal.SIGINT, signal.SIG_IGN)
    print("Worker init with id", os.getpid())
    env = env_fn()
    try:
        while True:
            #print("Waiting for command...")
            cmd, data = remote.recv()
            #print("Received command:", cmd)
            if cmd == 'step':
                    ob, reward, done, _, info = env.step(data)
                    remote.send((ob, reward, done, info))
            elif cmd == 'get_screen':
                #screen = env._get_observation()
                screen = env._get_observation().transpose((2, 0, 1))
                screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
                screen = torch.from_numpy(screen)
                screen = resize(screen).unsqueeze(0)
                remote.send(screen)
                #remote.send(screen.cpu().numpy())
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.action_space, env.observation_space))
            else:
                raise NotImplementedError
    finally:
        env.close()

class MultiprocessVectorEnv:
    def __init__(self, env_fns, resize, device):
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=worker, args=(work_remote, env_fn, resize, device))
                for (work_remote, env_fn) in zip(self.work_remotes, env_fns)
        ]
        for p in self.ps:
            p.start()
        self.last_obs = [None] * self.num_envs
        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()
        self.closed = False
        self.device = device

    def __del__(self):
        if not self.closed:
            self.close()


    def step(self, actions):
        self._assert_not_closed()
        #actions = [torch.Tensor(action).to(self.device) for action in actions]
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        self.last_obs, rews, dones, infos = zip(*results)
        return self.last_obs, rews, dones, infos
    
    def get_screen(self):
        for remote in self.remotes:
            remote.send(('get_screen', None))
        results = [remote.recv() for remote in self.remotes]
        screens = torch.cat(results,dim=0)
        #screens = torch.cat([torch.Tensor(result).to(self.device) for result in results],dim=0)
        return screens

    def reset(self, mask=None):
        self._assert_not_closed()
        if mask is None:
            mask = np.zeros(self.num_envs)
        for m, remote in zip(mask, self.remotes):
            if not m:
                remote.send(('reset', None))

        obs = [remote.recv() if not m else o for m, remote,
               o in zip(mask, self.remotes, self.last_obs)]
        self.last_obs = obs
        return obs

    def close(self):
        self._assert_not_closed()
        self.closed = True
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)

    def _assert_not_closed(self):
        assert not self.closed, "This env is already closed"
        
def make_env(idx, test, seed, capture_video, run_name, device):
    env = KukaDiverseObjectEnv(
        render_mode="rgb_array",
        renders=False, 
        isDiscrete=False, 
        removeHeightHack=False, 
        maxSteps=20
    )
    env.observation_space = spaces.Box(low=0., high=1., shape=(84, 84, 3), dtype=np.float32)
    env.action_space = spaces.Box(low=-1, high=1, shape=(5,1))
    
    # TODO not working with .get_screen() (AttributeError: accessing private attribute '_get_observation' is prohibited)
    if capture_video:
        if idx == 0:
            #env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda t: t % 10 == 0)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    
    env.np_random = seeding.np_random(seed)[0]
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)        
    return env

def make_batch_env(test, resize, num_envs, seed, capture_video, run_name, device=torch.device("cpu")):
    return MultiprocessVectorEnv(
        [functools.partial(make_env, idx, test, seed, capture_video, run_name, device) 
            for idx in range(num_envs)],
        resize,
        device
    )