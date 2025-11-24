import torch
import torch.nn as nn

x = torch.randn(1, 3, 6, 6, requires_grad=True)
print(x)
# in_channels: R, G, B
#out_channels: # of filters used in convolution
# kernel size: dimension of each kernel used in convolution
conv_layer = nn.Conv2d(in_channels = 3, out_channels = 2, kernel_size = 3, stride = 1, padding = 1)
output = conv_layer(x)
print(output)

class TwoLayerConvModel((torch.nn.Module)):
    def __init__(self):
        super(TwoLayerConvModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 2, kernel_size = 3, stride = 1, padding = 1)
        

class TwoLayerModel((torch.nn.Module)):
    def __init__(self):
        super(TwoLayerModel, self).__init__()

        self.lin1 = torch.nn.Linear(100, 80)
        self.activation1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(80, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self):
        Z1 = self.lin1(x)
        A1 = self.activation1(Z1)
        Z2 = self.lin2(A1)
        A2 = self.activation1(Z2)
        YPred = self.softmax(A2)
        return YPred
    