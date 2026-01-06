import torch
import torch.nn as nn

class vgg16model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            #block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            #block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            #block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size = 2, stride =2),
            #block 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            #block 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features = 7*7*512, out_features=4096),
            nn.ReLU(inplace= True),
            nn.Linear(in_features = 4096, out_features=4096),
            nn.ReLU(inplace= True),
            nn.Linear(in_features = 4096, out_features=1),
        )
        
    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
