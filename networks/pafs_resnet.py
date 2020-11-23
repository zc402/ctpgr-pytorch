import torch
import torch.nn as nn
import torchvision


class ResnetPAFs(nn.Module):
    """Top 10 Layers of VGG19 + Two conv layers"""
    def __init__(self):
        super(ResnetPAFs, self).__init__()
        self.net = torchvision.models.resnet50(pretrained=False)
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 14+11, kernel_size=1)
        )

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        # x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        # x = self.net.layer4(x)

        x = self.classifier(x)
        b1 = x[:, :14, ...]
        b2 = x[:, 14:, ...]
        return [b1], [b2], b1, b2
