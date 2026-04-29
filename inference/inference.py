import numpy as np
import torch
import tqdm

torch.manual_seed(42)

def predict(model, dataloader):

    outputs = []
    labels= []

    model.eval()
    with torch.no_grad():
        bar = tqdm.tqdm(dataloader, desc = 'Begin prediction..', colour='green')
        for data in bar:
            img, label = data
            outputs.append(torch.argmax(model(img), dim=1))
            labels.append(label)

    outputs = np.array(outputs)
    labels = np.array(labels)

    return outputs,labels

def accuracy(outputs, labels):

    mask = outputs == labels
    accuracy = np.sum(mask)/len(labels)
    return accuracy

def accuracy_per_class(outputs, labels):
    ans = [0] * 10
    norm = [0] * 10
    for i in range(len(outputs)):
        if outputs[i] == labels[i]:
            ans[outputs[i][0]] += 1
        norm[labels[i][0]] += 1

    ans = [ round(ans[i]/norm[i], 4) for i in range(0,10) ]

    return ans

