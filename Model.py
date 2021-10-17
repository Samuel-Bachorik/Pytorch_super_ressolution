import torch
import torch.nn as nn

class Super_ress_model(torch.nn.Module):
    def __init__(self, input_shape= (3, 256, 352), output_shape= (2, 256, 352)):
        super(Super_ress_model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers1 = [
            self.conv_layer(input_shape[0], 32, 7,1,3),
            self.conv_layer(32, 64, 3,1,1),
            self.conv_layer(64, 128, 3,1,1),
        ]
        self.upsample = torch.nn.Upsample(size=None, scale_factor=3, mode='nearest', align_corners=None)

        self.layers2 = [
            self.conv_layer(128, 128, 3,1,1),
            self.conv_layer(128,128, 3,1,1),
            self.conv_layer(128, 3, 3,1,1),
        ]

        self.model_layers1 = nn.Sequential(*self.layers1)
        self.model_layers1.to(self.device)

        self.model_layers2 = nn.Sequential(*self.layers2)
        self.model_layers2.to(self.device)

        print(self.model_layers2)
        print(self.model_layers1)


    def forward(self,x):
        layers_1 = self.model_layers1(x)
        upsample = self.upsample(layers_1)
        layers2 = self.model_layers2(upsample)

        return layers2


    def conv_layer(self, inputs, outputs,kernel, stride,padding):
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True))



