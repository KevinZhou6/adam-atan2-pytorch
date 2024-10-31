import os
import matplotlib.pyplot as plt
import torch 
import numpy as np

LABELS=['SGD','Adam','AdamAtan2']

def get_folder_path(use_pretrained=True):
    if use_pretrained:
        path='pretrained'
    else:
        path ='curve'
    return path 

def get_curve_data(use_pretrained=True,model='Resnet34'):
    folder_path = get_folder_path(use_pretrained)
    filenames =[name for name in os.listdir(folder_path) if name.startswith(model.lower())]
    paths =[os.path.join(folder_path,name) for name in filenames]
    
    keys = [name.split('-')[1] for name in filenames]
    return {keys:torch.load(fp) for key,fp in zip(keys,paths)}


def plot(use_pretrained=True,model='resnet18',optimizers=None,curve_type='train'):
    assert model in ['resnet34','resnet18','resnet50','resnet101','resnet152'],'Invalid model name: {}'.format(model)
    
    assert curve_type in ['train', 'test'], 'Invalid curve type: {}'.format(curve_type)
    assert all(_ in LABELS for _ in optimizers), 'Invalid optimizer'
    
    curve_data = get_curve_data(use_pretrained,model)
    plt.figure()
    plt.title('{} Accuracy for {} on CIFAR-100'.format(curve_type.capitalize(), model))
    plt.xlabel('Epoch')
    plt.ylabel('{} Accuracy %'.format(curve_type.capitalize()))
    if curve_type == 'train':
        plt.ylim(80, 101)
    else:
        plt.ylim(50, 81)

    for optim in optimizers:
        accuracies = np.array(curve_data[optim.lower()]['{}_acc'.format(curve_type)])
        plt.plot(accuracies, label=optim)

    plt.grid(ls='--')
    plt.legend()
    plt.show()
    plt.savefig('cifar100-{}-{}.png'.format(model, curve_type.capitalize()))
    
def main():
    # plot(use_pretrained=True, model='ResNet', optimizers=LABELS, curve_type='train')
    # plot(use_pretrained=True, model='ResNet', optimizers=LABELS, curve_type='test')

    plot(use_pretrained=True, model='resnet18', optimizers=LABELS, curve_type='train')
    plot(use_pretrained=True, model='resnet18', optimizers=LABELS, curve_type='test')

if __name__ == '__main__':
    main()