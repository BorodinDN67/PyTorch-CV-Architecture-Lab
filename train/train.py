import torch
import tqdm
from torch import nn
from torch import optim
from torch.optim import optimizer

torch.manual_seed(42)

class Trainer:
    def __init__(self, args, model, dataloader):
        self.device = args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss = torch.nn.CrossEntropyLoss()
        self.epochs = args['epochs']
        self.dataloader = dataloader
        self.lr = float(args['lr'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self):

        bar = tqdm.tqdm(range(self.epochs), desc='Training in progress', position=0)
        for t in bar:
            self.model.train()
            bar_train = tqdm.tqdm(self.dataloader, desc=f' Epoch: {t}', leave=False, position=1)
            for data in bar_train:
                input, target = data
                input = input.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                bar_train.set_postfix({'loss': loss.item()})
                if loss < 0.001: #заделка под раннюю остановку
                    break



