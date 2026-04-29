import torch.nn as nn
import torch.nn.functional as F
import torch
from openpyxl.styles.builtins import output
from utils.layers import SelfAttention, FeedForwardNetwork

class fc(nn.Module):
    #Просто какая то произвольная полносвязная сеть
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in,128 )
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, dim_out)

    def forward(self, x):
        x = x.reshape(x.size(0), 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,padding = 1 )
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.reshape(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class vit(nn.Module):
    def __init__(self, patch_dim, dim_in, dim_out, d_model, in_channels=1):
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.count_of_patches = int( (dim_in // self.patch_dim)**2 )
        self.d_model = int(d_model)
        self.in_channels = in_channels
        self.d_model = d_model

        self.patches = nn.Conv2d(1, self.d_model, stride=self.patch_dim, kernel_size=self.patch_dim, bias = False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.SelfAttention1 = SelfAttention(in_features= self.d_model, d_model = self.d_model )
        self.FFN1 = FeedForwardNetwork(in_features= self.d_model,hidden_dim= d_model * 4,out_features = self.d_model)
        self.LayerNorm1 = nn.LayerNorm(d_model)
        self.SelfAttention2 = SelfAttention(in_features=  self.d_model, d_model = self.d_model )
        self.FFN2 = FeedForwardNetwork(in_features= self.d_model,hidden_dim= d_model * 4,out_features = self.d_model)
        self.LayerNorm2 = nn.LayerNorm(d_model)
        self.SelfAttention3 = SelfAttention(in_features= self.d_model, d_model = self.d_model )
        self.LayerNorm3 = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, dim_out)
        )


    def forward(self, x):
        x = self.patches(x)

        x = x.flatten(start_dim = 2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim = 1)

        x = self.LayerNorm1(self.SelfAttention1(x))
        x = self.FFN1(x)
        x = self.LayerNorm2(self.SelfAttention2(x))
        x = self.FFN2(x)
        x = self.LayerNorm3(self.SelfAttention3(x))
        x = self.classifier(x[:,0,:])
        return x

def get_model(args):
    if args['model_name'] == 'fc':
        return fc( dim_in = args['dim_in'],dim_out =  args['dim_out'])
    elif args['model_name'] == 'cnn':
        return CNN()
    elif args['model_name'] == 'vit':
        return vit(patch_dim = args['patch'], dim_in = args['dim_in'],dim_out = args['dim_out'] , d_model = args['d_model'])
    else:
        print('Model not supported (model)')
        exit()