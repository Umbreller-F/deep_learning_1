from numpy import append
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def my_fc(in_features, out_features):
    blk = []
    blk.append(nn.Linear(in_features,out_features))
    blk.append(nn.BatchNorm1d(out_features))
    blk.append(nn.ReLU())
    blk.append(nn.Dropout(0.15))
    return nn.Sequential(*blk)


def my_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.Dropout(0.15))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return nn.Sequential(*blk)


def my_flatten():
    blk = []
    blk.append(nn.Flatten())
    return nn.Sequential(*blk)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv0 = nn.Conv2d(3, 32, 3, stride = 1, padding = 1) #(64*64*3) -> (64*64*32)
        self.bn0 = nn.BatchNorm2d(32)
        
        ######################## task 1.1 ##########################
        self.blk1 = my_block(1, 32, 64) #(64*64*32) -> (32*32*64)
        self.blk2 = my_block(1, 64, 128) #(32*32*64) -> (16*16*128)
        self.blk3 = my_block(1, 128, 256) #(16*16*128) -> (8*8*256)
        self.blk4 = my_block(1, 256, 512) #(8*8*256) -> (4*4*512)
        # self.fc1 = nn.Linear(4*4*512, 4*512)
        self.flat = my_flatten()
        self.fc1 = my_fc(4*4*512, 4*512)
        # self.fc2 = nn.Linear(4*512, 512)
        self.fc2 = my_fc(4*512, 512)
        self.fc3 = nn.Linear(512, 200)
        ########################    END   ##########################
        self.dropout = nn.Dropout(0.15)
                
    def forward(self, input):
        x = F.relu(self.bn0(self.dropout(self.conv0(input))))
        
        ######################## task 1.2 ##########################
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        # x = x.view(-1, 4*4*512)
        x = self.flat(x)
        # x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc1(x)
        # x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc2(x)
        x = self.fc3(x)
        # Tips x = x.view(-1, 3*3*512)
        ########################    END   ##########################

        return x