from torchaudio.models.wav2vec2.components import FeedForward

from models.models import vit
import torch

x = torch.randn(1,1,100)
y = torch.Tensor([[[0,1,1,1,1]]])


print(y)
print(x.shape)
print(y.shape, y.shape[-1])
S = FeedForwardNetwork(input_dim = x.shape[-1], output_dim = 10, hidden_dim = 512)
print(S(x))