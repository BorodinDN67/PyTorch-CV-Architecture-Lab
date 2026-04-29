import sys
import os
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder, DatasetFolder
import numpy as np


#Опишем верхнеуровнево загрузчики данных, используемые в опыте
# Train/ val / test - Тест у нас идет отдельно, будем для упрощения (в плане слияния и перераспределения данных)
# И усложенения задачи (в плане меньшего числа данных)
# Набирать val набор из test

# Для разных архитектур подберем разные аугментации. Валидацию оставим для всех одинаковую


path = './data/mnist/train'

transform_fc = T.Compose([
    T.Resize((28,28)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

transform_CNN = T.Compose([
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

transform_VT = T.Compose([
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

transform_ResNet = T.Compose([
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

transform_test = T.Compose([
    T.Resize((28,28)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])


class CustomDataset(Dataset):
    def __init__(self, path, alpha = 0.1, transform = None):
        self.dataset = []
        self.labels = []
        self.alpha = alpha
        self.path = path
        self.transform = transform

        for fnmae in os.listdir(path):
            if fnmae.endswith('.png'):
                label = fnmae.split('_')[0]
                self.labels.append(label)
                self.dataset.append(fnmae)

        self.classes = sorted(list(set(self.labels)))
        self.classes_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.labels = [self.classes_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = Image.open(self.path + '/' + self.dataset[idx])
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)

        return img, label



def get_dataloader(args, model = 'fc', mode = 'train'):
    transform = None
    if mode == 'test':
        transform = transform_test
    else:
        if model == 'fc':
            transform = transform_fc
        elif model == 'cnn':
            transform = transform_CNN
        elif model == 'vit':
            transform = transform_VT
        elif model == 'ResNet':
            transform = transform_ResNet
        else:
            print('model not supported (dataset)')
            sys.exit()


    dataset = CustomDataset(path = args.get('path','./data/mnist/train' ),transform =  transform, alpha = args.get('alpha', 0.1) )

    class_count = np.bincount(dataset.labels)
    class_weight = 1.0 / torch.Tensor(class_count).float()

    sample_weight = class_weight[dataset.labels] #ЧЕРНАЯ МАГИЯ

    sampler = WeightedRandomSampler( weights = sample_weight, num_samples= len(sample_weight), replacement = True)


    dataloader = DataLoader(
        dataset,
        batch_size = args.get('batch_size', 1),
        shuffle = False,
        num_workers = args.get('num_workers', 0),
        sampler = sampler,
    )

    return dataloader

