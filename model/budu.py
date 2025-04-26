import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt



# ResNet-50 Encoder
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]

class SegmentationDecoder(nn.Module):
    def __init__(self, num_classes_drivable=2, num_classes_lane=2):
        super(SegmentationDecoder, self).__init__()
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        
        # Add upsampling to match input size
        self.up_final = nn.Upsample(size=(360, 640), mode='bilinear', align_corners=False)
        
        self.drivable_head = nn.Conv2d(32, num_classes_drivable, kernel_size=1)
        self.lane_head = nn.Conv2d(32, num_classes_lane, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.up1(x)  # [batch_size, 256, 24, 40]
        x = self.conv2(x)
        x = self.up2(x)  # [batch_size, 64, 48, 80]
        x = self.conv3(x)
        x = self.up3(x)  # [batch_size, 32, 96, 160]
        x = self.up_final(x)  # [batch_size, 32, 360, 640]
        drivable_out = self.drivable_head(x)  # [batch_size, 2, 360, 640]
        lane_out = self.lane_head(x)  # [batch_size, 2, 360, 640]
        return drivable_out, lane_out

# Full Segmentation Model
class SegmentationModel(nn.Module):
    def __init__(self, num_classes_drivable=2, num_classes_lane=2):
        super(SegmentationModel, self).__init__()
        self.encoder = ResNet50Encoder(pretrained=True)
        self.decoder = SegmentationDecoder(num_classes_drivable, num_classes_lane)

    def forward(self, x):
        features = self.encoder(x)
        drivable_out, lane_out = self.decoder(features[-1])  # Use only layer4
        return drivable_out, lane_out
