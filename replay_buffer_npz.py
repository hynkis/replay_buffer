import numpy as np
import torch
import os
import random

class ReplayBufferNumPy():
    """
    Allocate data as numpy & save buffer as npz file.

    """
    def __init__(self, args, device):
        self.device = device
        self.max_capacity = args.max_capacity
        self.idx = 0 # index for put data
        self.curr_size = 0
        self.obs_img = np.empty((self.max_capacity, 3, 256, 256), dtype=np.float32)
        self.actions = np.empty((self.max_capacity, args.action_dim), dtype=np.float32)
        self.rewards = np.empty((self.max_capacity, 1), dtype=np.float32) 
        self.next_obs_img = np.empty((self.max_capacity, 3, 256, 256), dtype=np.float32)
        self.dones = np.empty((self.max_capacity, 1), dtype=bool)

    def put(self, transition):
        # append numpy data
        obs_img, action, reward, next_obs_img, done = transition
        self.obs_img[self.idx] = obs_img
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs_img[self.idx] = next_obs_img
        self.dones[self.idx] = done
        # circular indexing
        self.idx = (self.idx + 1) % self.max_capacity
        # size of data
        self.curr_size = min(self.curr_size + 1, self.max_capacity) # current size of data in buffer
    
    def sample(self, n):
        idxs = np.random.randint(0, self.curr_size, size=n)

        obs_img = self.obs_img[idxs]           # [batch, 3, 256, 256]; [[(C,H,W)], ..., [(C,H,W)]]
        action = self.actions[idxs]            # [batch, dim]
        reward = self.rewards[idxs]            # [batch, 1]
        next_obs_img = self.next_obs_img[idxs] # [batch, 3, 256, 256];
        done = np.array(self.dones[idxs], dtype=np.float32) # bool to float32; # [batch, 1]
        
        return torch.tensor(obs_img, dtype=torch.float32).to(self.device), \
               torch.tensor(action, dtype=torch.float32).to(self.device), \
               torch.tensor(reward, dtype=torch.float32).to(self.device), \
               torch.tensor(next_obs_img, dtype=torch.float32).to(self.device), \
               torch.tensor(done, dtype=torch.float32).to(self.device)
    
    def size(self):
        return self.curr_size

    def save(self, save_dir, datetime, n_epi):
        file_path = datetime + 'ReplayBuffer_' + str(n_epi) + '.npz'
        memory_file_path = os.path.join(save_dir, file_path)
        np.savez_compressed(memory_file_path,
                            n_epi = n_epi,
                            max_capacity = self.max_capacity,
                            idx = self.idx,
                            curr_size = self.curr_size,
                            obs_img = self.obs_img,
                            actions = self.actions,
                            rewards = self.rewards,
                            next_obs_img = self.next_obs_img,
                            dones = self.dones,
                            )
        print(f'{memory_file_path} save success!!!')

    def load(self, memory_file_path):
        if os.path.isfile(memory_file_path):
            print("Start to load buffer!")
            loaded_buffer = np.load(memory_file_path, allow_pickle=True)

            self.max_capacity = loaded_buffer['max_capacity']
            self.idx = loaded_buffer['idx']
            self.curr_size = loaded_buffer['curr_size']
            self.obs_img = loaded_buffer['obs_img']
            self.actions = loaded_buffer['actions']
            self.rewards = loaded_buffer['rewards']
            self.next_obs_img = loaded_buffer['next_obs_img']
            self.dones = loaded_buffer['dones']

            loaded_buffer.close()
            print(f'Replay buffer is loaded. {memory_file_path}')
        else:
            print("There is NO ReplayBuffer. Getting initial experiences will be going on...")

