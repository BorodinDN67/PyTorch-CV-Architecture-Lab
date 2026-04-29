if __name__ == '__main__' :
    import torch
    import yaml
    import os
    import datetime
    from data.data import get_dataloader, CustomDataset
    from train.train import Trainer
    from models.models import fc, get_model
    import matplotlib.pyplot as plt

    MODE = 'train' #правится в коде т.к используется только для написания сценариев
    MODEL = 'vit'


    def logg_res(MODEL,train_args,model_args, num, path  ):
        day = datetime.datetime.now()
        name_file = './logs/'+f'{MODEL}' + '/exp_' + MODEL + '_' + day.strftime('%d_%m_%Y__%H_%M') + '.txt'

        with open(name_file, 'x', encoding='utf-8') as file:
            arr = [f'================Модель: {MODEL}======================\n',
                    f'Параметры обучения: num_workers: {train_args["num_workers"]}\n',
                    f'batch_size: {train_args["batch_size"]}\n',
                    f'lr: {train_args["lr"]}\n',
                    f'epoch: {train_args["epochs"]}\n',
                    f'Параметры модели: {model_args.items()}\n',
                    f'Номер эксперимента {num}\n',
                    f'Путь к весам: {path}\n']
            for line in arr:
                file.write(line)

    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

        train_args = config['train']
        model_args = config['model'][MODEL]


    model = get_model(model_args)
    dataloader = get_dataloader(train_args, model = model_args['model_name'], mode = MODE)
    trainer = Trainer(dataloader = dataloader,model =  model, args = train_args)
    trainer.fit()

    N = len([i for i in os.listdir(train_args.get('path_saved', './saved/') +  f'{MODEL}/')])

    torch.save(model.state_dict(),train_args.get('path_saved', './saved/') +  f'{MODEL}/' + f'{N+1}'  )
    logg_res(MODEL,train_args,model_args, N+1,train_args.get('path_saved', './saved/') +  f'{MODEL}/' + f'{N+1}' )




