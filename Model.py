import torch
import torch.nn as nn

class Super_ress_model(torch.nn.Module):
    def __init__(self, input_shape= (3, 128, 128), output_shape= (3, 384, 384)):
        super(Super_ress_model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers_1 = [
            self.conv_layer(input_shape[0], 32, 7,1,3),
            self.conv_layer(32, 64, 3,1,1),
            self.conv_layer(64, 128, 3,1,1),
        ]
        self.upsample = torch.nn.Upsample(size=None, scale_factor=4, mode='nearest', align_corners=None)

        self.layers_2 = [
            self.conv_layer(128, 64, 3,1,1),
            self.conv_layer(64,64, 3,1,1),
            self.conv_layer(64, output_shape[0], 3,1,1),
        ]

        for i in range(len(self.layers_2)):
            if hasattr(self.layers_2[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_2[i].weight)
                torch.nn.init.zeros_(self.layers_2[i].bias)

        for i in range(len(self.layers_1)):
            if hasattr(self.layers_1[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_1[i].weight)
                torch.nn.init.zeros_(self.layers_1[i].bias)

        self.model_layers1 = nn.Sequential(*self.layers_1)
        self.model_layers1.to(self.device)

        self.model_layers2 = nn.Sequential(*self.layers_2)
        self.model_layers2.to(self.device)

        print(self.model_layers2)
        print(self.model_layers1)


    def forward(self,x):
        x_bypass = self.upsample(x)
        layers_1 = self.model_layers1(x)
        upsample = self.upsample(layers_1)
        layers2 = self.model_layers2(upsample)

        return layers2 + x_bypass


    def conv_layer(self, inputs, outputs,kernel, stride,padding):
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True))
