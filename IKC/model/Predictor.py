import torch
from torch import nn

class Predictor(nn.Module):
    def __init__(self, img_channels=3, kernel_channels=10, hidden_channels=64):
        super(Predictor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=hidden_channels,
                      kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                      kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                      kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=kernel_channels,
                      kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.globalPooling = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        h0 = self.globalPooling(conv4)
        h0 = h0.view(h0.size()[:2])
        
        return h0