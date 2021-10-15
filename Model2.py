import torch
import torch.nn as nn

class Super_ress_model(torch.nn.Module):
    def __init__(self, input_shape= (3, 256, 352), output_shape= (2, 256, 352)):
        super(Super_ress_model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layer1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1)
        self.activation1 = nn.ReLU()

        self.layer2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.activation2 = nn.ReLU()

        self.upsample = torch.nn.Upsample(size=None, scale_factor=3, mode='nearest', align_corners=None)

        self.layer3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.activation3 = nn.ReLU()

        self.layer4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.activation4 = nn.ReLU()


    def forward(self,x):
        x = self.layer1(x)
        x = self.activation1(x)

        x = self.layer2(x)
        x = self.activation2(x)

        x = self.upsample(x)

        x = self.layer3(x)
        x = self.activation3(x)

        x = self.layer4(x)
        x = self.activation4(x)

        return x



