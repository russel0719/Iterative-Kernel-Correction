import torch
from torch import nn

class Corrector(nn.Module):
    def __init__(self, img_channels=3, kernel_channels=10, hidden_channels=64):
        super(Corrector, self).__init__()
        
        self.hidden_channels = hidden_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=hidden_channels,
                      kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                      kernel_size=(5,5), stride=(2,2), padding=(2,2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                      kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                      kernel_size=(5,5), stride=(2,2), padding=(2,2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                      kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels*2, out_channels=hidden_channels*2,
                      kernel_size=(1,1), stride=(1,1), padding=(0,0)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels*2, out_channels=hidden_channels,
                      kernel_size=(1,1), stride=(1,1), padding=(0,0)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv8 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=kernel_channels,
            kernel_size=(1,1), stride=(1,1), padding=(0,0)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=kernel_channels, out_features=hidden_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.globalPooling = nn.AdaptiveAvgPool2d([1, 1])
        
    def forward(self, x, h):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        fc1 = self.fc1(h)
        fc2 = self.fc2(fc1)
        b, c, height, width = conv5.size()
        stretched_h = fc2.view((-1, self.hidden_channels, 1, 1)).expand((-1, self.hidden_channels, height, width))
        concat = torch.cat((conv5, stretched_h), dim=1)
        conv6 = self.conv6(concat)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)
        delta_h = self.globalPooling(conv8)
        delta_h = delta_h.view(delta_h.size()[:2])
        
        return delta_h