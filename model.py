import torch
import torch.nn as nn
from torchsummary import summary
from MABlock import SpatialCNNAtt, ChannelAtt
import torch.nn.functional as F

class ASCat(nn.Module):
    def __init__(self, channel=64):
        super(ASCat, self).__init__()
        self.conv1 = nn.Conv2d(channel*2, channel, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channel)
    def forward(self, x, y):
        z = torch.cat([x,y], dim=1)
        return F.relu(self.bn1(self.conv1(z)))

class ASLayer(nn.Module):
    def __init__(self, channel=64, SK_size = 3):
        super(ASLayer, self).__init__()
        self.spaCnnAtt = SpatialCNNAtt(in_channels=channel, SK_size=SK_size)
        self.chaAtt = ChannelAtt(in_channels=channel)
        self.conv1 = nn.Conv2d(channel*2, channel, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channel)
    def forward(self, x ,y):
        f = self.chaAtt(x) * self.spaCnnAtt(y)
        z = torch.cat([f*x, f*y], dim=1)
        return F.relu(self.bn1(self.conv1(z)))

class ASNet(nn.Module):
    def __init__(self):
        super(ASNet, self).__init__()
        channel = 64
        #spectral_num = 8
        spectral_num = 4
        self.deconv = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4, padding=2, bias=True)
        self.upsample = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4, padding=2, bias=True)
        self.msl1 = ASLayer(channel = channel,SK_size = 3)
        self.msl2 = ASLayer(channel = channel,SK_size = 5)
        self.msl3 = ASLayer(channel = channel,SK_size = 7)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=channel, kernel_size=5, padding=2)
        self.ascat1 = ASCat(channel=channel)
        self.ascat2 = ASCat(channel=channel)
    def forward(self, x, y):  
        skip = self.upsample(x)
        # forward propagation
        x = self.relu(self.deconv(x))
        x = self.relu(self.conv1(x))
        y = self.relu(self.conv3(y))
        x1 = self.msl1(x, y)
        x2 = self.msl2(x, y)
        x3 = self.msl3(x, y)
        x = self.ascat2(self.ascat1(x1,x2),x3)
        x = self.conv2(x)
        return x + skip

