import torch
import torch.nn as nn
import torch.nn.functional as f

class Memory_Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False):
        super(Memory_Net, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
    def forward(self, x, initial_state=None):
        if initial_state is None:
            h0 = torch.zeros((1, x.size(0), self.gru.hidden_size)).to(x.device)
        else:
            h0 = initial_state
        out, h = self.gru(x, h0)
        return out, h