from models.MFCAB import ResidualBlock
import torch.nn as nn



class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


class MFCANN(nn.Module):
    def __init__(self, in_channels, out_channels, intra_reduction_radio=1, inter_reduction_radio=4, n_filters=32, kernel_sizes=[10, 20, 40], bottleneck_channels=32, activation=nn.ReLU(), device=None):
        super(MFCANN, self).__init__()

        self.residual1 = ResidualBlock(in_channels=in_channels, n_filters=n_filters, kernel_sizes=kernel_sizes, bottleneck_channels=bottleneck_channels,
                                       intra_reduction_radio=intra_reduction_radio, inter_reduction_radio=inter_reduction_radio, activation=activation, device=device)
        self.residual2 = ResidualBlock(n_filters * 5, n_filters=n_filters, kernel_sizes=kernel_sizes, bottleneck_channels=bottleneck_channels,
                                       intra_reduction_radio=intra_reduction_radio, inter_reduction_radio=inter_reduction_radio, activation=activation, device=device)
        self.classes_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=n_filters * 5 * 1),
            nn.Linear(in_features=5 * n_filters * 1, out_features=out_channels, device=device)
        )


    def forward(self, X):
        X = self.residual1(X)
        X = self.residual2(X)
        output = self.classes_head(X)
        return output
