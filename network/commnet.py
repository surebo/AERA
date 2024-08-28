import torch
import torch.nn as nn
import torch.nn.functional as f


# input obs of all agents，output probability distribution of all agents
class CommNet(nn.Module):
    def __init__(self, input_shape, args):
        super(CommNet, self).__init__()
        self.encoding = nn.Linear(input_shape, args.rnn_hidden_dim)  
        self.f_obs = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  
        self.f_comm = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  
        self.decoding = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.args = args
        self.input_shape = input_shape

    def forward(self, obs, hidden_state):

        obs_encoding = torch.sigmoid(self.encoding(obs))  # .reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        h_out = self.f_obs(obs_encoding, h_in)

        for k in range(self.args.k):  
            if k == 0: 
                h = h_out
                c = torch.zeros_like(h)
            else:
                h = h.reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)

                c = h.reshape(-1, 1, self.args.n_agents * self.args.rnn_hidden_dim)
                c = c.repeat(1, self.args.n_agents, 1)  
                mask = (1 - torch.eye(self.args.n_agents)) 
                mask = mask.view(-1, 1).repeat(1, self.args.rnn_hidden_dim).view(self.args.n_agents, -1)  # (n_agents, n_agents * rnn_hidden_dim))
                if self.args.cuda:
                    mask = mask.cuda()
                c = c * mask.unsqueeze(0)
                c = c.reshape(-1, self.args.n_agents, self.args.n_agents, self.args.rnn_hidden_dim)
                c = c.mean(dim=-2)  # (episode_num * max_episode_len, n_agents, rnn_hidden_dim)
                h = h.reshape(-1, self.args.rnn_hidden_dim)
                c = c.reshape(-1, self.args.rnn_hidden_dim)
            h = self.f_comm(c, h)
        weights = self.decoding(h)

        return weights, h_out

