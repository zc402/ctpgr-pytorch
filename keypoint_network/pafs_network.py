import numpy as np
import torch
import torch.nn as nn
import torchvision


class Vgg19Top10(nn.Module):
    """Top 10 Layers of VGG19 + Two conv layers"""
    def __init__(self):
        super(Vgg19Top10, self).__init__()
        self.top10_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.top10_model(x)
        return x


class StageOne(nn.Module):
    """The stage 1 of PAFs Network"""
    def __init__(self, in_channels, num_classes):
        super(StageOne, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

class StageT(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(StageT, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class PAFsNetwork(nn.Module):
    """
    The network of paper "Part Affinity Fields"
    returns:
        b1_stages (list): All the branch 1 outputs
        b2_stages (list): All the branch 2 outputs
        b1: Prediction output for branch 1
        b2: Prediction output for branch 2
    """

    def __init__(self, b1_classes, b2_classes):
        super(PAFsNetwork, self).__init__()
        self.features = Vgg19Top10()
        features_out_ch = 128  # num outputs of vgg19Top10
        self.s1b1 = StageOne(in_channels=features_out_ch, num_classes=b1_classes)  # s1b1: Stage 1, Branch 1.
        self.s1b2 = StageOne(in_channels=features_out_ch, num_classes=b2_classes)
        num_stage_out_classes = b1_classes + b2_classes + features_out_ch
        self.s2b1 = StageT(in_channels=num_stage_out_classes, num_classes=b1_classes)
        self.s2b2 = StageT(in_channels=num_stage_out_classes, num_classes=b2_classes)
        self.s3b1 = StageT(in_channels=num_stage_out_classes, num_classes=b1_classes)
        self.s3b2 = StageT(in_channels=num_stage_out_classes, num_classes=b2_classes)

    def forward(self, x):
        x0 = self.features(x)
        x1_1 = self.s1b1(x0)
        x1_2 = self.s1b2(x0)
        x1 = torch.cat((x1_1, x1_2, x0), 1)

        x2_1 = self.s2b1(x1)
        x2_2 = self.s2b2(x1)
        x2 = torch.cat((x2_1, x2_2, x0), 1)

        x3_1 = self.s3b1(x2)
        x3_2 = self.s3b2(x2)

        b1_stages = [x1_1, x2_1, x3_1]  # All branch 1 outputs
        b2_stages = [x1_2, x2_2, x3_2]  # All branch 2 outputs
        b1 = x3_1  # Branch 1 final output
        b2 = x3_2  # Branch 2 final output
        return b1_stages, b2_stages, b1, b2


class PAFsLoss(nn.Module):
    """
    pred: shape (N, Stage, C, H, W)
    """
    def forward(self, pred_b1, pred_b2, gt_pcm, gt_paf):
        assert len(pred_b1.shape) == 5  # (N, Stage, C, H, W)
        assert len(gt_pcm.shape) == 5
        pred = torch.cat((pred_b1, pred_b2), dim=2)
        gt = torch.cat((gt_pcm, gt_paf), dim=2)
        square = (pred - gt) ** 2
        sum_over_one_batch = torch.sum(square, dim=(3, 4))  # Shape (N)
        mean = sum_over_one_batch.mean()
        return mean