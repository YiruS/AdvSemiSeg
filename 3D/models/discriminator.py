from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class ConvDiscNet(nn.Module):
    def __init__(self, input_dim):
        super(ConvDiscNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 16, 1)
        self.fc = nn.Linear(16, 1)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.transpose(2, 1) # (BxNxC) -> (BxCxN)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.transpose(2, 1)  # (BxNxC')
        x = self.fc(x) # (BxNx1)
        x = x.squeeze(2)
        return x

class PointNetDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(PointNetDiscriminator, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1) # 64
        self.conv2 = torch.nn.Conv1d(64, 64, 1) # 64
        self.conv3 = torch.nn.Conv1d(64, 128, 1) # 128
        self.conv4 = torch.nn.Conv1d(128, 128, 1)  # 128
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)  # 1024

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc1 = torch.nn.Linear(1024, 512) # 128
        self.fc2 = torch.nn.Linear(512, 256) # 1024
        self.fc3 = torch.nn.Linear(256, 64) # 64
        self.fc4 = torch.nn.Linear(64, 64) # 64
        self.fc5 = torch.nn.Linear(64, self.output_dim)

        # self.bn6 = nn.BatchNorm1d(512)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.bn8 = nn.BatchNorm1d(64)
        # self.bn9 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.transpose(2,1) # BxCxN
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))

        x = x.traspose(2,1) # BxNxC
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(self.fc3(x))
        x = self.dropout(self.fc4(x))
        x = self.fc5(x)

        return x