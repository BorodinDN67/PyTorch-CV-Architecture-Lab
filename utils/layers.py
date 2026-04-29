import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_features, d_model = 512):
        super().__init__()
        self.Q = nn.Linear(in_features, d_model,bias=False)
        self.K = nn.Linear(in_features, d_model,bias=False)
        self.V = nn.Linear(in_features, d_model,bias=False)
        self.d_model = d_model
    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.d_model ** 0.5

        attention = F.softmax(attention, dim=-1)

        attention = torch.matmul(attention,V)




        return attention



class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
