import torch
import torch.nn as nn
import torch.nn.functional as F


class HourglassBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HourglassBlock, self).__init__()
        self.down1 = self._conv_block(in_channels, out_channels)
        self.down2 = self._conv_block(out_channels, out_channels)
        self.up1 = self._conv_block(out_channels, out_channels)
        self.up2 = self._conv_block(out_channels, out_channels)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        up1 = self.up1(down2)
        up2 = self.up2(up1)
        return up2 + down1 + down2


class HourglassNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(HourglassNet, self).__init__()
        self.initial_conv = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hourglass1 = HourglassBlock(64, 128)
        self.hourglass2 = HourglassBlock(128, 256)

        self.final_conv = nn.Conv2d(256, n_classes, kernel_size=3, stride=1, padding=1)  # Output channel = 1

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.hourglass1(x)
        x = self.hourglass2(x)
        x = self.final_conv(x)
        return x

    def predict(self, x):
        x = self.initial_conv(x)
        x = self.hourglass1(x)
        x = self.hourglass2(x)
        return x


if __name__ == '__main__':
    # Instantiate the model and create a dummy input tensor of size (batch_size, 3, 350, 250)
    model = HourglassNet()
    dummy_input = torch.randn(1, 3, 350, 250)  # Batch size = 1, channels = 3, height = 350, width = 250

    # Forward pass
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should print torch.Size([1, 1, 350, 250])
