import numpy as np
import torch
import os
import random

class ReplayBufferDisc():
    """
    Save transition data as a npz file in the disc, rather than RAM (for efficient memory usage).

    """
    def __init__(self, args, save_dir, device):
        self.mini_buffer = []
        self.max_capacity = args.buffer_size
        self.save_dir = save_dir
        self.device = device

    def put(self, transition):
        self.mini_buffer.append(transition)
    
    def sample(self, n):
        """
        - Sample data list
        - Load data in batch
        - Stack data
        - Return as tensor data

        """
        # Sample data list
        file_list = os.listdir(self.save_dir)
        mini_batch_path = random.sample(file_list, n)

        # Load data in batch
        s_img_lst, s_obs_lst = [], []
        a_lst, r_lst, done_mask_lst = [], [], []
        s_img_prime_lst, s_obs_prime_lst = [], []
        
        # Stack data
        for file_path in mini_batch_path:
            transition = np.load(os.path.join(self.save_dir, file_path), mmap_mode='r')
            # - convert uint8 image to tensor image
            s_img = transition["s_img"] # [C, H, W], numpy
            s_img_prime = transition["s_img_prime"] # [C, H, W], numpy
            # - preprocess
            s_img = self.normalize_img(s_img)
            s_img_prime = self.normalize_img(s_img_prime)
            # - stack data
            s_img_lst.append(s_img)                # [batch, C, H, W]; [[C,H,W], [C,H,W], ..., [C,H,W]]
            s_obs_lst.append(transition["s_obs"])  # [batch, dim]; [[dim], [dim], ..., [dim]]
            a_lst.append(transition["a"])   # [batch, dim]
            r_lst.append([transition["r"]]) # [batch, 1]
            s_img_prime_lst.append(s_img_prime)               # [batch, C, H, W];
            s_obs_prime_lst.append(transition["s_obs_prime"]) # [batch, dim];
            done_mask = 1.0 if transition["done"] else 0.0 
            done_mask_lst.append([done_mask]) # [batch, 1]
        
        # - Return as tensor data
        return torch.tensor(np.float32(s_img_lst), dtype=torch.float).to(self.device), \
               torch.tensor(np.float32(s_obs_lst), dtype=torch.float).to(self.device), \
               torch.tensor(a_lst, dtype=torch.float).to(self.device), \
               torch.tensor(np.float32(r_lst), dtype=torch.float).to(self.device), \
               torch.tensor(np.float32(s_img_prime_lst), dtype=torch.float).to(self.device), \
               torch.tensor(np.float32(s_obs_prime_lst), dtype=torch.float).to(self.device), \
               torch.tensor(done_mask_lst, dtype=torch.float).to(self.device)
               
    def save_episode_transitions(self, n_epi):
        """
        - save list of transitions as npz files during current episode
        """
        # Save transition data in current episode
        for transition in self.mini_buffer:
            s_img, s_obs, a, r, s_img_prime, s_obs_prime, done, n_epi, episode_step = transition
            data_name = str(n_epi).zfill(4) + "_" + str(episode_step).zfill(4) + ".npz"
            data_path = os.path.join(self.save_dir, data_name)
            np.savez_compressed(data_path,
                                s_img = s_img,
                                s_obs = s_obs,
                                a = a,
                                r = r,
                                s_img_prime = s_img_prime,
                                s_obs_prime = s_obs_prime,
                                done = done,
                                n_epi = n_epi,
                                episode_step = episode_step)
        
        # Check file list in save_dir path
        data_list = os.listdir(self.save_dir)
        if (len(data_list) > self.max_capacity):
            # remove past data until max capacity
            num_data_remove = int(len(data_list) - self.max_capacity)
            data_list.sort()
            for i in range(num_data_remove):
                data_remove_path = os.path.join(self.save_dir, data_list[i])
                if os.path.isfile(data_remove_path):
                    os.remove(data_remove_path)
                    print(f"{data_list[i]} file is removed")
            # update length of data
            data_list = os.listdir(self.save_dir)
        print(f' (Episode: {n_epi}, Steps: {len(self.mini_buffer)} ) transition data are saved!!! Current number of data : {len(data_list)}')

        # Clear memory
        self.mini_buffer = []

    def size(self):
        file_list = os.listdir(self.save_dir)
        return len(file_list)

    def normalize_img(self, cv_img):
        norm_np_img = cv_img / 255. # cv image to normalized torch tensor
        norm_np_img -= 0.5
        norm_np_img *= 2.

        return norm_np_img

