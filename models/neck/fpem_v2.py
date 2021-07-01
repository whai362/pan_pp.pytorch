import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..utils import Conv_BN_ReLU

class FPEM_v2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPEM_v2, self).__init__()
        planes = out_channels
        self.dwconv3_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv1_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.smooth_layer2_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv3_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.smooth_layer3_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv4_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.smooth_layer4_2 = Conv_BN_ReLU(planes, planes)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, f1, f2, f3, f4):
        f3_ = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        f2_ = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3_, f2)))
        f1_ = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2_, f1)))

        f2_ = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2_, f1_)))
        f3_ = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3_, f2_)))
        f4_ = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3_)))

        f1 = f1 + f1_
        f2 = f2 + f2_
        f3 = f3 + f3_
        f4 = f4 + f4_

        return f1, f2, f3, f4
