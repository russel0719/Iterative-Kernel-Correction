import torch
from torch import nn

class SFTMD(nn.Module):
    def __init__(self, img_channels=3, kernel_channels=10, hidden_channels=64, num_block=16, scale=2):
        super(SFTMD, self).__init__()
        
        self.num_block = num_block
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=hidden_channels,
                      kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                      kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.sftconv1 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels + kernel_channels, out_channels=32,
                      kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=hidden_channels,
                      kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        )
        self.sftconv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels + kernel_channels, out_channels=32,
                      kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=hidden_channels,
                      kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        )
        self.sftsigmoid = nn.Sigmoid()
        
        self.residualconv1 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(3,3), stride=(1,1), padding=(1,1)
        )
        self.residualconv2 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(3,3), stride=(1,1), padding=(1,1)
        )
        
        self.conv3 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(3,3), stride=(1,1), padding=(1,1)
        )
        self.conv4 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=img_channels,
            kernel_size=(3,3), stride=(1,1), padding=(1,1)
        )
        if scale == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * scale,
                          kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * scale,
                          kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * scale**2,
                          kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
    def SFTLayer(self, F, H):
        concat = torch.cat((F, H), dim=1)
        sftconv1 = self.sftconv1(concat)
        gamma = self.sftsigmoid(sftconv1)
        beta = self.sftconv2(concat)
        return gamma * F + beta
    
    def residualBlock(self, F, H):
        sft1 = self.SFTLayer(F, H)
        conv1 = self.residualconv1(sft1)
        sft2 = self.SFTLayer(conv1, H)
        conv2 = self.residualconv2(sft2)
        return conv2 + F
    
    def forward(self, x, h):
        b, c, height, width = x.size()
        b_h, c_h = h.size()
        stretched_h = h.view((b_h, c_h, 1, 1)).expand((b_h, c_h, height, width))
        H = stretched_h.reshape(b, -1, height, width)
        
        conv1 = self.conv1(x)
        F = self.conv2(conv1)
        
        for i in range(self.num_block):
            F = self.residualBlock(F, H)
        
        sft = self.SFTLayer(F, H)
        conv3 = self.conv3(sft)
        upscale = self.upscale(conv3)
        out = self.conv4(upscale)
        
        return out