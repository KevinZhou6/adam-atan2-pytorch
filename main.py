"""Train CIFAR100 with PyTorch."""

from __future__ import annotations,print_function

import torch 
import torch.optim as optim 
import torch.backends.cudnn as cudnn 
import torchvision
import torchvision.transforms as transforms 

import os 
import argparse 

from Models import * 
from adam_atan2_pytorch import * 

def get_parser():
    parser = argparse.ArgumentParser(description="Pytorch CIFAR100, Training")
    parser.add_argument("--model",default="resnet34",type=str,help="model",choices=['resnet34','resnet18','resnet50','resnet101','resnet152'])
    parser.add_argument("--optim",default="adam_atan2",type=str,help="optimizer",choices=['sgd','adam','adam_atan2'])
    parser.add_argument("--lr",default=0.05,type=float,help='learning rate')
    parser.add_argument("--beta1",default=0.9,type=float,help ="Adam coefficients beta_1")
    parser.add_argument("--beta2",default=0.99,type=float,help = "Adam coefficients beta_2")
    parser.add_argument("--momentum",default =0.9,type=float,help = "momentum term")
    parser.add_argument("--resume",'-r',action="store_true",help="resume from checkpoint")
    parser.add_argument("--weight_decay",default=5e-4,type=float,help="weight decay for opt")
    
    return parser 

def build_dataset():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./dataset', train=True, download=True,
                                             transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,
                                               num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./dataset', train=False, download=True,
                                            transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader

def get_ckpt_name(dataset="cifar100",model="resnet",optimizer="adam_atan2",lr=0.05,momentum=0.9,beta1=0.9,beta2=0.99):
    name={
        'adam':'lr{}-beatas{}-{}'.format(lr,beta1,beta2),
        'sgd':'lr{}-momentum{}'.format(lr,momentum),
        'adam_atan2':'lr{}-betas{}-{}'.format(lr,beta1,beta2)
    }[optimizer]
    
    return '{}-{}-{}'.format(model,optimizer,name)

def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint',ckpt_name)
    assert os.path.isdir('checkpoint'),'Error:no checkpoint directory found!'
    assert os.path.exists(path),'Error: checkpoint{} not found'.format(ckpt_name)
    return torch.load(ckpt_name)

def build_model(args,device,ckpt=None):
    print("==> Building model..")
    net ={
        'resnet18':ResNet18,
        'resnet34':ResNet34,
        'resnet50':ResNet50,
        'resnet101':ResNet101,
        'resnet152':ResNet152,
        
        
    }[args.model]()
    net = net.to(device)
    
    if device =="cuda":
        net =torch.nn.DataParallel(net)
        cudnn.benchmarks=True 
    
    if ckpt:
        net.load_state_dict(ckpt['net'])
    
    return net

def create_optimizer(args,model_params):
    if args.optim=="sgd":
        return optim.SGD(model_params,args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.optim=="adam":
        return optim.Adam(model_params,args.lr,betas=(args.beta1,args.beta2),weight_decay=args.weight_decay)
    elif args.optim=="adam_atan2":
        return AdamAtan2(model_params,args.lr,betas=(args.beta1,args.beta2),weight_decay=args.weight_decay)

def train(net,epoch,device,data_loader,optimizer,criterion):
    print(f"Epoch :{epoch}")
    
    net.train()
    
    train_loss=0
    correct=0
    total =0
    
    for batch_idx,(inputs,targets) in enumerate(data_loader):
        inputs,targets = inputs.to(device),targets.to(device)
        optimizer.zero_grad()
        out = net(inputs)
        loss =criterion(out,targets)
        loss.backward()
        optimizer.step()
        
        train_loss +=loss.item()
        _,predicted = out.max(1)
        total +=targets.size(0)
        correct +=predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print('train acc %.3f' % accuracy)
    
    return accuracy

def test(net,device,data_loader,criterion):
    net.eval()
    test_loss=0
    total =0
    correct=0
    with torch.no_grad():
        for batch_id,(inputs,targets) in enumerate(data_loader):
            inputs,targets = inputs.to(device),targets.to(device)
            out = net(inputs)
            
            test_loss +=criterion(out,targets).item()
            _,predicts = out.max(1)
            total+=targets.size(0)
            correct += predicts.eq(targets).sum().item()
    accuracy = 100. * correct / total
    print(' test acc %.3f' % accuracy)
    return accuracy

def main():
    parser =get_parser()
    args = parser.parse_args()
    train_loader,test_loader = build_dataset()
    device ="cuda:3" if torch.cuda.is_available() else "cpu"
    
    ckpt_name = get_ckpt_name(model=args.model,optimizer=args.optim,lr=args.lr,
                              momentum=args.momentum,beta1=args.beta1, beta2=args.beta2,)
    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']
    else:
        ckpt =None 
        best_acc=0.
        start_epoch=-1 
    model =build_model(args,device,ckpt=ckpt)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args,model.parameters())
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,[150,225],gamma=0.1,last_epoch=start_epoch
    )
    
    train_accuracies=[]
    test_accuracies=[]
    
    for epoch in range(start_epoch+1,300):
        scheduler.step()
        train_acc = train(model,epoch,device,train_loader,optimizer,criterion)
        test_acc = test(model,device,test_loader,criterion)
        
        if test_acc > best_acc:
            print("Saving..")
            state={
                "net":model.state_dict(),
                "acc":test_acc,
                "epoch":epoch,
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(state,os.path.join('checkpoint',ckpt_name))
            best_acc = test_acc 
            
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        if not os.path.isdir('curve'):
            os.mkdir('curve')
        torch.save({"train_acc":train_accuracies,"test_acc":test_accuracies},os.path.join('curve',ckpt_name))
    
    print(f"---Model: {args.model} -----Optimizer:{args.optim} ----- BestAcc:{best_acc}")
    
    
    
if __name__ =="__main__":
    main()        
    