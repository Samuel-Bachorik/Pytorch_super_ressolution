import torch
import torch.nn as nn

class ResidualBlock(torch.nn.Module):
    def __init__(self, filters,kernel_size = 3, init_gain = 0.1):
        super(ResidualBlock, self).__init__()

        self.conv0 = nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.bn0   = nn.BatchNorm2d(filters)
        self.act0  = nn.ReLU()

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.bn1   = nn.BatchNorm2d(filters)
        self.act1  = nn.ReLU()

        torch.nn.init.xavier_uniform_(self.conv0.weight)
        torch.nn.init.zeros_(self.conv0.bias)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)

    def forward(self, x):
        y = self.conv0(x)
        y = self.bn0(y)
        y = self.act0(y)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act1(y + x)

        return y


class Super_ress_model(torch.nn.Module):
    def __init__(self, input_shape= (3, 128, 128), output_shape= (3, 384, 384)):
        super(Super_ress_model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=9 // 2)
        self.block0 = ResidualBlock(64,3)
        self.block1 = ResidualBlock(64,3)
        self.block2 = ResidualBlock(64,3)
        self.block3 = ResidualBlock(64,3)
        self.block4 = ResidualBlock(64,3)
        self.block5 = ResidualBlock(64, 3)
        self.block6 = ResidualBlock(64, 3)
        self.block7 = ResidualBlock(64, 3)

        self.upsample = torch.nn.Upsample(size=None, scale_factor=4, mode='nearest', align_corners=None)

        self.block8 = ResidualBlock(64,3)
        self.block9 = ResidualBlock(64,3)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3 // 2)
        self.act1 = nn.ReLU()

        self.output = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=3 // 2)

    def forward(self,x):
        x_bypass = self.upsample(x)
        x = self.input(x)
        x = self.block0.forward(x)
        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)
        x = self.block4.forward(x)
        x = self.block5.forward(x)
        x = self.block6.forward(x)
        x = self.block7.forward(x)
        x = self.upsample(x)
        x = self.block8.forward(x)
        x = self.block9.forward(x)
        x = self.conv1(x)
        x = self.act1(x)

        x = self.output(x)

        return x + x_bypass
