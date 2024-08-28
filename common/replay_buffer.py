import numpy as np
import threading
import json
from network.replay_RNN import Memory_Net

class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        
        
        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        's': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'r': np.empty([self.size, self.episode_limit, 1]),
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'padded': np.empty([self.size, self.episode_limit, 1]),
                        'terminated': np.empty([self.size, self.episode_limit, 1])
                        }
        
        
        
        if self.args.alg == 'maven':
            self.buffers['z'] = np.empty([self.size, self.args.noise_dim])
        # thread lock
        self.lock = threading.Lock()

        # store the episode
    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']
            if self.args.alg == 'maven':
                self.buffers['z'][idxs] = episode_batch['z']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    
    #TODO algorithm2
    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
    
    def mutual_information(self):
        s = self.buffers['s'][:self.current_size].reshape(-1, self.args.state_shape)
        s_counts = np.unique(s, axis=0, return_counts=True)[1] 
        s_probs = s_counts / np.sum(s_counts) 
        
        w_counts = np.unique(self.buffers['w'][:self.current_size], return_counts=True)[1]
        w_probs = w_counts / np.sum(w_counts)
    
        sw = np.concatenate((s, self.buffers['w'][:self.current_size]), axis=1)
        sw_counts = np.unique(sw, axis=0, return_counts=True)[1]
        sw_probs = sw_counts / np.sum(sw_counts)
    

        mi = 0
        for i in range(len(s_probs)): 
            for j in range(len(w_probs)):  
                idx = i * len(w_probs) + j
                if sw_probs[idx] > 0:
                    mi += sw_probs[idx] * np.log2(sw_probs[idx] / (s_probs[i] * w_probs[j]))
    
        return mi
    


class ReplayBuffer_with_RNN(ReplayBuffer):
    def __init__(self, args):
        super().__init__(args)
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.buffers['h'] = np.empty([self.size, self.episode_limit, self.n_agents, self.rnn_hidden_dim])
        self.buffers['h_next'] = np.empty([self.size, self.episode_limit, self.n_agents, self.rnn_hidden_dim])
        self.get_emmbedding = Memory_Net(input_size=self.obs_shape, hidden_size=self.rnn_hidden_dim, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False)
        
        

    def sample(self, batch_size,k):

        temp_buffer = {}       
        for i in range(batch_size):  
            idx = np.random.randint(0, self.current_size, k) 
            temp_buffer_k = {}      
            for key in self.buffers.keys():     
                temp_buffer_k[key] = self.buffers[key][idx]  

            o_k = temp_buffer_k['o']       
            o_k = np.reshape(o_k, (k*self.episode_limit, self.n_agents, self.obs_shape)) 
            h0 = np.zeros((self.n_agents, self.rnn_hidden_dim))
            h0 = np.tile(h0, (k, 1))  
            _, h = self.get_emmbedding(o_k, initial_state=h0)  
            h = np.reshape(h, (k, self.n_agents, self.rnn_hidden_dim))  

            temp_buffer_i = {}     
            for key in self.buffers.keys():
                if key in ['o', 'u', 'r', 'terminated', 'avail_u', 'u_onehot', 'padded']:  
                    temp_buffer_i[key] = np.reshape(temp_buffer_k[key], (k*self.episode_limit, -1))
                elif key in ['s', 's_next']:
                    temp_buffer_i[key] = np.reshape(temp_buffer_k[key], (k*self.episode_limit, -1))
                elif key in ['o_next', 'avail_u_next']:
                    temp_buffer_i[key] = np.reshape(temp_buffer_k[key], (k*self.episode_limit, self.n_agents, -1))
                elif key in ['h', 'h_next']:
                    temp_buffer_i[key] = np.reshape(h, (k*self.episode_limit, self.n_agents, self.rnn_hidden_dim))

            for key in temp_buffer_i.keys():
                if i == 0:
                    temp_buffer[key] = temp_buffer_i[key]
                else:
                    temp_buffer[key] = np.concatenate([temp_buffer[key], temp_buffer_i[key]], axis=0)
        return temp_buffer
