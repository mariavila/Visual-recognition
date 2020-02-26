import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(1, 1), stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=(1, 1), stride=1)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 8)

    def forward(self, x):

        x = self.bn1(self.maxpool(torch.relu(self.conv1(x))))
        x = torch.relu(self.conv2(x))
        x = self.bn2(self.maxpool(torch.relu(self.conv3(x))))
        x = torch.relu(self.conv4(x))
        x = self.bn3(self.maxpool(torch.relu(self.conv5(x))))
        x = torch.relu(self.conv6(x))
        x = F.avg_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1]))
        x = self.fc(x.squeeze())

        return x

