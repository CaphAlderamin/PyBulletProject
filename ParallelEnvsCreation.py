
import numpy as np
import os

import functools
import multiprocessing as mp
from multiprocessing import Pipe
from multiprocessing import Process
import signal

import torch

from gym import spaces
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv  
import pybullet as p

from Attendants import get_resize
resize = get_resize()

def worker(remote, env_fn):
    # Ignore CTRL+C in the worker process
    print("Worker init with id", os.getpid())
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = env_fn()
    try:
        while True:
            #print("Waiting for command...")
            cmd, data = remote.recv()
            #print("Received command:", cmd)
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == 'get_screen':
                #screen = env._get_observation()
                screen = env._get_observation().transpose((2, 0, 1))
                screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
                screen = torch.from_numpy(screen)
                screen = resize(screen).unsqueeze(0)
                remote.send(screen)
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
    def __init__(self, env_fns):
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=worker, args=(work_remote, env_fn))
                for (work_remote, env_fn) in zip(self.work_remotes, env_fns)
        ]
        for p in self.ps:
            p.start()
        #print("Before receiving spaces...")
        self.last_obs = [None] * self.num_envs
        #print("Before sending get_spaces command...")
        self.remotes[0].send(('get_spaces', None))
        #print("Sent get_spaces message")
        self.action_space, self.observation_space = self.remotes[0].recv()
        #print("Received spaces")
        self.closed = False

    def __del__(self):
        if not self.closed:
            self.close()


    def step(self, actions):
        self._assert_not_closed()
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
        
def make_env(idx, test):
    env = KukaDiverseObjectEnv(
        renders=False, 
        isDiscrete=False, 
        removeHeightHack=False, 
        maxSteps=20
    )
    env.observation_space = spaces.Box(low=0., high=1., shape=(84, 84, 3), dtype=np.float32)
    env.action_space = spaces.Box(low=-1, high=1, shape=(5,1))
    #print("\nEnv", idx, "was created.")
    return env

def make_batch_env(test):
    return MultiprocessVectorEnv([
            functools.partial(make_env, idx, test) 
                for idx in range(mp.cpu_count()*2)
                #for idx in range(2)
        ])