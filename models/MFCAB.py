import math

import torch
import torch.nn as nn
from torch.nn import ModuleList
from .Attention import *


def pass_through(X):
    return X


class MFCAB(nn.Module):
    def __init__(self, in_channels, n_filters, intra_reduction_radio, inter_reduction_radio, kernel_sizes=[10, 20, 40], bottleneck_channels=32, activation=nn.ReLU(), device=None):
        super(MFCAB, self).__init__()
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                device=device
            )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        self.conv_from_bottleneck_1 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False,
            device=device
        )
        self.conv_from_bottleneck_2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False,
            device=device
        )
        self.conv_from_bottleneck_3 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False,
            device=device
        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_from_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            device=device
        )
        self.conv_from_avgpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            device=device
        )
        self.batch_norm = nn.BatchNorm1d(num_features=5 * n_filters)
        self.activation = activation

        self.intra_module_attention1 = IntraModuleAttention(n_filters, reduction_ratio=intra_reduction_radio)
        self.intra_module_attention2 = IntraModuleAttention(n_filters, reduction_ratio=intra_reduction_radio)
        self.intra_module_attention3 = IntraModuleAttention(n_filters, reduction_ratio=intra_reduction_radio)
        self.intra_module_attention4 = IntraModuleAttention(n_filters, reduction_ratio=intra_reduction_radio)
        self.intra_module_attention5 = IntraModuleAttention(n_filters, reduction_ratio=intra_reduction_radio)

        self.inter_module_attention = InterModuleAttention(module_num=5, in_channels=n_filters, device=device, reduction_ratio=inter_reduction_radio)


    def forward(self, X):
        Z_bottleneck = self.bottleneck(X)
        Z_maxpool = self.max_pool(X)
        Z_avgpool = self.avg_pool(X)

        Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
        Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
        Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
        Z4 = self.conv_from_maxpool(Z_maxpool)
        Z5 = self.conv_from_avgpool(Z_avgpool)

        Z = torch.cat([Z1, Z2, Z3, Z4, Z5], axis=1)
        Z = self.batch_norm(Z)
        Z = self.activation(Z)

        split_tensors = torch.split(Z, split_size_or_sections=[32, 32, 32, 32, 32], dim=1)
        Z1 = split_tensors[0].cuda()
        Z2 = split_tensors[1].cuda()
        Z3 = split_tensors[2].cuda()
        Z4 = split_tensors[3].cuda()
        Z5 = split_tensors[4].cuda()

        Z_list = [Z1, Z2, Z3, Z4, Z5]
        output_vector = self.inter_module_attention(Z_list)

        Z1 = self.intra_module_attention1(Z1)
        Z2 = self.intra_module_attention2(Z2)
        Z3 = self.intra_module_attention3(Z3)
        Z4 = self.intra_module_attention4(Z4)
        Z5 = self.intra_module_attention5(Z5)

        stacked_Z = torch.stack([Z1, Z2, Z3, Z4, Z5], dim=1).cuda()
        stacked_Z = stacked_Z * output_vector

        Z1 = stacked_Z[:, 0]
        Z2 = stacked_Z[:, 1]
        Z3 = stacked_Z[:, 2]
        Z4 = stacked_Z[:, 3]
        Z5 = stacked_Z[:, 4]

        Z = torch.cat([Z1, Z2, Z3, Z4, Z5], axis=1)

        return Z


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, intra_reduction_radio, inter_reduction_radio, n_filters=32, kernel_sizes=[10, 20, 40], bottleneck_channels=32, activation=nn.ReLU(), device=None):
        super(ResidualBlock, self).__init__()
        self.activation = activation
        self.mfcab_1 = MFCAB(
            in_channels=in_channels,
            n_filters=n_filters,
            intra_reduction_radio=intra_reduction_radio,
            inter_reduction_radio=inter_reduction_radio,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            device=device,
        )
        self.mfcab_2 = MFCAB(
            in_channels=5 * n_filters,
            n_filters=n_filters,
            intra_reduction_radio=intra_reduction_radio,
            inter_reduction_radio=inter_reduction_radio,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            device=device,
        )
        self.mfcab_3 = MFCAB(
            in_channels=5 * n_filters,
            n_filters=n_filters,
            intra_reduction_radio=intra_reduction_radio,
            inter_reduction_radio=inter_reduction_radio,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            device=device,
        )

        self.residual = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=5 * n_filters,
                kernel_size=1,
                stride=1,
                padding=0,
                device=device
            ),
            nn.BatchNorm1d(
                num_features=5 * n_filters
            )
        )

    def forward(self, X):
        Z = self.mfcab_1(X)
        Z = self.mfcab_2(Z)
        Z = self.mfcab_3(Z)
        Z = Z + self.residual(X)
        Z = self.activation(Z)
        return Z



