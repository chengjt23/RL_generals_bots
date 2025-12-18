import numpy as np
import torch
from multiprocessing import Process, Pipe
from generals.envs import PettingZooGenerals
from generals.agents import RandomAgent


def worker(remote, parent_remote, env_config):
    parent_remote.close()
    
    env = PettingZooGenerals(agents=["Agent", "Opponent"], render_mode=None)
    
    while True:
        cmd, data = remote.recv()
        
        if cmd == 'step':
            agent_action, opponent_action = data
            actions_dict = {"Agent": agent_action, "Opponent": opponent_action}
            obs_dict, rewards_dict, terminated, truncated, info = env.step(actions_dict)
            done = terminated or truncated
            
            if done:
                final_rewards = rewards_dict.copy()
                obs_dict, _ = env.reset()
                remote.send((obs_dict, final_rewards, True, info))
            else:
                remote.send((obs_dict, rewards_dict, False, info))
        
        elif cmd == 'reset':
            obs_dict, info = env.reset()
            remote.send((obs_dict, info))
        
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        
        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


class ParallelEnvs:
    def __init__(self, n_envs, env_config=None):
        self.n_envs = n_envs
        self.env_config = env_config or {}
        
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            p = Process(target=worker, args=(work_remote, remote, self.env_config))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()
    
    def step_async(self, agent_actions, opponent_actions):
        for remote, agent_action, opponent_action in zip(self.remotes, agent_actions, opponent_actions):
            remote.send(('step', (agent_action, opponent_action)))
    
    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        obs_dicts, rewards_dicts, dones, infos = zip(*results)
        return obs_dicts, rewards_dicts, dones, infos
    
    def step(self, agent_actions, opponent_actions):
        self.step_async(agent_actions, opponent_actions)
        return self.step_wait()
    
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs_dicts, infos = zip(*results)
        return obs_dicts, infos
    
    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()

