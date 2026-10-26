import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.layer2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)
        self.act = nn.GeLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x.T)
        return x # l2(act(f1(x))) 