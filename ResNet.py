import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ResNetLayer(nn.Module):
    def __init__(self, in_feature_maps, out_feature_maps, downsample = True):
        super(ResNetLayer, self).__init__()

        self.stride = 2 if downsample == True else 1
        self.conv0 = nn.Conv2d(in_feature_maps, out_feature_maps, 3, stride = self.stride, padding = 1)
        self.bn0 = nn.BatchNorm2d(out_feature_maps)
        self.conv1 = nn.Conv2d(out_feature_maps, out_feature_maps, 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_feature_maps)

        self.skipconn_cnn = nn.Conv2d(in_feature_maps, out_feature_maps, kernel_size=1, stride=self.stride, padding = 0)
        self.skipconn_bn = nn.BatchNorm2d(out_feature_maps)

    def forward(self, input):
        ######################## task 2.1 ##########################
        output = self.conv0(input)
        output = self.bn0(output)
        output = F.relu(output)
        output = self.conv1(output)
        output = self.bn1(output)
        output = F.relu(output + self.skipconn_bn(self.skipconn_cnn(input)))
        return output
        ########################    END   ##########################


def ResNetBlock(in_feature_maps, out_feature_maps, layers):
    blk = []
    blk.append(ResNetLayer(in_feature_maps, out_feature_maps))
    for _ in range(layers-1):
        blk.append(ResNetLayer(out_feature_maps, out_feature_maps, downsample=False))
    return nn.Sequential(*blk)


def my_flatten():
    blk = []
    blk.append(nn.Flatten())
    return nn.Sequential(*blk)


def my_fc(in_features, out_features):
    blk = []
    blk.append(nn.Linear(in_features,out_features))
    blk.append(nn.BatchNorm1d(out_features))
    blk.append(nn.ReLU())
    blk.append(nn.Dropout(0.15))
    return nn.Sequential(*blk)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.conv0 = nn.Conv2d(3, 64, 3, stride = 1, padding = 1) #(64*64*3) -> (64*64*64)
        self.bn0 = nn.BatchNorm2d(64)
        
        ######################## task 2.2 ##########################
        self.blk1 = ResNetBlock(64, 128, 3)
        self.blk2 = ResNetBlock(128, 256, 4)
        self.blk3 = ResNetBlock(256, 512, 6)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.blk4 = ResNetLayer(512, 512, downsample=False)
        self.flat = my_flatten()
        self.fc1 = my_fc(4*4*512, 4*512)
        # self.fc2 = nn.Linear(4*512, 512)
        self.fc2 = my_fc(4*512, 512)
        self.fc3 = nn.Linear(512, 200)
        ########################    END   ##########################

        self.dropout = nn.Dropout(0.15)

    def forward(self, input):
        x = F.relu(self.bn0(self.dropout(self.conv0(input))))
        
        ######################## task 2.3 ##########################
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.avgpool(x)
        x = self.blk4(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        ########################    END   ##########################

        return x