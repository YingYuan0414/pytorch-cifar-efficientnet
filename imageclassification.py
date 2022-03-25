from dataset import CIFAR10_4x
from evaluation import evaluation

from model import Net  # this should be implemented by yourself

import math
import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from evaluation import evaluation

from torchvision import transforms
from PIL import Image


def set_seed(seed):
    seed = int(seed)
    if seed < 0 or seed > (2**32 - 1):
        raise ValueError("Seed must be between 0 and 2**32 - 1")
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def train(mod):
    ##############################################################################
    #                  TODO: You need to complete the code here                  #
    ##############################################################################
    # YOUR CODE HERE
    loss_tracking = []
    acc = []
    print("Begin training!")
    for epoch in range(500):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                loss_tracking.append(running_loss / 200)
                with open('loss1.json', 'a') as f:
                    json.dump(str(running_loss / 200), f)
                running_loss = 0.0
        if epoch % 10 == 9:
            torch.save(net, os.path.join(model_dir, (str)(epoch) + '.pth'))
            acc_valid = evaluation(net, validloader, device)
            acc_train = evaluation(net, trainloader, device)
            with open('acc_valid9.txt', 'a') as f:
                json.dump(str(acc_valid), f)
            with open('acc_train9.txt', 'a') as f:
                json.dump(str(acc_train), f)

        scheduler.step()

    print("Finished Training")
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="DL Coding2")
    parser.add_argument('--model', default='resnet18')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    set_seed(16)

    data_root_dir = '.'

    transform = transforms.Compose([
        # transforms.Resize((16, 16)),
        transforms.RandomChoice([
            transforms.RandomResizedCrop(128),
            transforms.RandomAffine(15),
            transforms.RandomHorizontalFlip(p=0.4)
        ]),
        transforms.RandomGrayscale(p=0.1),
        # transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([125 / 255, 124 / 255, 115 / 255],
                            [60 / 255, 59 / 255, 64 / 255])
    ])

    trainset = CIFAR10_4x(root=data_root_dir,
                        split="train", transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    validset = CIFAR10_4x(root=data_root_dir,
                        split='valid', transform=transform)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=128, shuffle=False, num_workers=8)

    if args.model == 'resnext50':
        from models import resnext
        net = resnext.resnext50()
    if args.model == 'densenet121':
        from models import densenet
        net = densenet.densenet121()
    if args.model == 'googlenet':
        from models import googlenet
        net = googlenet.googlenet()
    if args.model == 'inceptionv3':
        from models import inceptionv3
        net = inceptionv3.inceptionv3()
    if args.model == 'seresnet34':
        from models import senet
        net = senet.seresnet34()
    if args.model == 'nasnet':
        from models import nasnet
        net = nasnet.nasnet()
    if args.model == 'resnet18':
        net = Net()
    if args.model == 'resnetv1':
        from models import resnetv1
        net = resnetv1.resnet_v1()
    if args.model == 'resnetv2':
        from models import resnetv1
        net = resnetv1.resnet_v2()
    if args.model == 'hcgnet':
        from models import HCGNet_CIFAR
        net = HCGNet_CIFAR.HCGNet_A2(10)
    if args.model == 'effnet':
        from models import effnet
        net = effnet.EfficientNet.from_name('efficientnet-b0')

    print("number of trained parameters: %d" % (
        sum([param.nelement() for param in net.parameters() if param.requires_grad])))
    print("number of total parameters: %d" %
        (sum([param.nelement() for param in net.parameters()])))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3) # weight_decay=5e-5
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(75, 90), gamma=0.4, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=0, last_epoch=-1)

    net.to(device)

    model_dir = './experiments/' + (str)(args.model) + '/10'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(net, os.path.join(model_dir, 'cifar10_4x_0.pth'))

    # check the model size
    os.system(' '.join(['du', '-h', os.path.join(model_dir, 'cifar10_4x_0.pth')]))



    train(args.model)