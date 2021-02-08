"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/2/8 16:56:01
# @File    : train_model.py
# @Software: PyCharm
-------------------------------------
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
from torch.optim import lr_scheduler
from torchvision import models, transforms

use_gpu = torch.cuda.is_available()

# device check
device = torch.device("cuda:0" if use_gpu else "cpu")
print('Running device: {}'.format(device))

# training settings
batch_size = 32
LR = 0.001

train_set = dset.ImageFolder(root='../images_data/train_images', transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
]))

validation_set = dset.ImageFolder(root='../images_data/validation_images',
                                  transform=transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_set,
                                                batch_size=batch_size,
                                                shuffle=False)

# load the pretrained model
vgg16 = models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16-397923af.pth"))
print(vgg16.classifier[6].out_features)

# freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False

# newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1]  # remove last layer
features.extend([nn.Linear(num_features, 11)])  # 11 species
features.extend([nn.LogSoftmax(1)])

vgg16.classifier = nn.Sequential(*features)  # replace the model classifier
vgg16.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(vgg16.parameters(), lr=LR, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


def train(vgg, optimizer, epoch, num_epochs=10):
    loss_over_time = []
    vgg.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = vgg(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch * num_epochs + batch_idx, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.tolist()))

            loss_over_time.append(loss.tolist())

    return loss_over_time


def test(vgg):
    vgg.eval()
    test_loss = 0
    correct = 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = vgg(data)
        # sum up batch loss
        # test_loss += F.nll_loss(output, target, size_average=False).data[0]
        test_loss += F.nll_loss(output, target, size_average=False).tolist()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(validation_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))

    return correct


if __name__ == '__main__':
    loss_curve = []
    for i in range(5):
        loss_curve.extend(train(vgg16, optimizer_ft, i))
        torch.save(vgg16, 'model.pb')

    # plot the loss function curve
    plt.plot(loss_curve)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('loss_curve.png')
    plt.show()

    test(vgg16)
    torch.save(vgg16, 'model.pb')
