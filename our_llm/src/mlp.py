import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.ffn_mult * config.n_embd

        self.fc_in = nn.Linear(config.n_embd, hidden)
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(hidden, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.activation(x)
        x = self.fc_out(x)
        return self.dropout(x)