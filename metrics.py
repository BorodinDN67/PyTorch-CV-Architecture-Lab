import torch
import yaml
import os
from data.data import get_dataloader, CustomDataset
from models.models import fc, get_model
from inference.inference import accuracy, predict,accuracy_per_class


MODE = 'test' #правится в коде т.к используется только мной для написания сценариев
MODEL = 'vit' #выбор модели, лучше не забыть или все упадет
VERSION = None # None - последняя по умолчанию


result = dict()

with open('config.yaml', 'r') as stream:
    config = yaml.safe_load(stream)

    train_args = config['train']
    model_args = config['model'][MODEL]
    test_args = config['test']


model = get_model(model_args)

if VERSION is None:
    VERSION = len([i for i in os.listdir(train_args.get('path_saved', './saved/') + f'{MODEL}/')])

model.load_state_dict(torch.load(train_args.get('path_saved', './saved/') + f'{MODEL}/' + f'{VERSION}'))

dataloader = get_dataloader(test_args, model = model_args['model_name'], mode = MODE)

predict_m, label = predict(dataloader = dataloader, model = model)

# print(predict_m.shape, label.shape)
print(f'MODEL: {MODEL}')
print('Some examples: ')
print('predict')
print(*predict_m[:15,:].reshape(1,-1))
print('true class:')
print(*label[:15,:].reshape(1,-1))
print()
print( f'accuracy: {accuracy(predict_m, label)}')
print()
print('accuracy per class: ')
res = list(accuracy_per_class(predict_m, label))
for i in range(0,10):
    print(f'Class {i}: {res[i]}')





