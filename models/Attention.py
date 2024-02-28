import math
import torch
import torch.nn as nn
from . import covariance


class IntraModuleAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=1):
        super(IntraModuleAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = torch.mean(x, dim=2)
        y = self.fc(y)
        y = y.unsqueeze(2)
        return x * y


class InterModuleAttention(nn.Module):
    def __init__(self, module_num, in_channels, device, reduction_ratio=4):
        super(InterModuleAttention, self).__init__()

        self.conv_list = nn.ModuleList()
        for i in range(module_num):
            self.conv_list.append(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=in_channels // reduction_ratio,
                          kernel_size=1,
                          stride=1,
                          bias=False,
                          device=device)
            )

        self.covmat_dim = module_num
        self.row_bn = nn.BatchNorm2d(self.covmat_dim)

        self.row_conv_group = nn.Conv2d(
            self.covmat_dim, 4 * self.covmat_dim,
            kernel_size=(self.covmat_dim, 1),
            groups=self.covmat_dim, bias=True)
        self.fc_adapt_channels = nn.Conv2d(
            4 * self.covmat_dim, self.covmat_dim,
            kernel_size=1, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, stacked_tensor):
        for index, layer in enumerate(self.conv_list):
            stacked_tensor[index] = layer(stacked_tensor[index])
        stacked_tensor = torch.stack(stacked_tensor, dim=1).cuda()

        out = covariance.CovpoolLayer(stacked_tensor)

        out = out.view(out.size(0), out.size(1), out.size(2), 1).contiguous()  # Nxdxdx1

        out = self.row_bn(out)
        out = self.row_conv_group(out) # Nx4dx1x1

        out = self.fc_adapt_channels(out) # Nxdx1x1
        out = self.sigmoid(out) # Nxdx1x1

        return out
