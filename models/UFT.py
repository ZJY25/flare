import torch
import torch.nn as nn
from utils.SmallWaveTransform import WavePool,WaveUnpool
from thop import clever_format
from utils import network_parameters

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Downsample(nn.Module):
    def __init__(self, kernel_size=4, padding=1, stride=2, in_chans=3, out_chans=32):
        super().__init__()
        self.down =nn.Conv2d(in_chans, in_chans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv=nn.Conv2d(in_chans,out_chans,kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.down(x)
        x=self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self,in_chans=3, out_chans=32):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chans, in_chans, kernel_size=2, stride=2)
        self.conv=nn.Conv2d(in_chans,out_chans,kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.up(x)
        x=self.conv(x)
        return x


class SpatialAttentionWeightBlock(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim//8, 1),
            act_layer(),
            nn.Conv2d(dim//8, dim, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.spatial_att(x)

class ChannelAttentionWeightBlock(nn.Module):
    def __init__(self, act_layer=nn.GELU):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        size_1 =3
        size_2 =5
        self.channelConv1 = nn.Conv1d(1, 1, size_1, padding=size_1//2)
        self.channelConv2 = nn.Conv1d(1, 1, kernel_size=size_2, padding=size_2//2)
        self.act = act_layer()

    def forward(self, x):
        x = self.avg_pool(self.act(x))
        x = self.channelConv1(x.squeeze(-1).transpose(-1, -2))
        x = self.act(x)
        x = self.channelConv2(x)
        x = x.transpose(-1, -2).unsqueeze(-1)
        return x


class BasicBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = dim
        self.act = nn.GELU()
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.fc2 =nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        temp=x
        x = self.fc1(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x+temp



class DynamicFusionBlock_3in(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.avg_op = nn.AdaptiveAvgPool2d((1, 1))
        self.convBlock=nn.Sequential(
            nn.Conv2d(3*dim, 3*dim // ratio, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(3*dim // ratio, 3*dim // ratio, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(3*dim // ratio, 3*dim, 1, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, x1, x2, x3):
        out1 = self.avg_op(x1)
        out2 = self.avg_op(x2)
        out3 = self.avg_op(x3)
        out = torch.cat([out1, out2, out3], dim=1)
        out=self.convBlock(out)
        w1, w2, w3 = torch.chunk(out, 3, dim=1)
        x = x1 * w1 + x2 * w2 + x3 * w3
        return x


class DynamicFusionBlock_2in(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.avg_op = nn.AdaptiveAvgPool2d((1, 1))
        self.convBlock = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // ratio, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(2 * dim // ratio, 2 * dim // ratio, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(2 * dim // ratio, 2 * dim, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        out1 = self.avg_op(x1)
        out2 = self.avg_op(x2)
        out = torch.cat([out1, out2], dim=1)
        out = self.convBlock(out)
        w1, w2 = torch.chunk(out, 2, dim=1)
        x = x1 * w1 + x2 * w2
        return x

class CooperateAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cab = ChannelAttentionWeightBlock()
        self.sab = SpatialAttentionWeightBlock(2*dim)
        self.conv_high=DynamicFusionBlock_2in(dim)
        self.conv_low=DynamicFusionBlock_2in(dim)

    def forward(self, high,low):
        total=torch.cat([high,low],dim=1)

        sweight = self.sab(total)
        total = total*sweight
        cweight = self.cab(total)
        total=total+cweight
        high_result,low_result=torch.chunk(total,2,dim=1)
        high=self.conv_high(high,high_result)
        low=self.conv_low(low,low_result)
        return high,low

class ResUNet(nn.Module):
    def __init__(self, dim=32):

        super().__init__()
        # self.input_pro = IOpro(in_chans=3, out_chans=dim)
        # self.output_pro=IOpro(in_chans=dim,out_chans=3)
        self.fcblock1=BasicBlock(dim)
        self.fcblock2 = BasicBlock(2 * dim)
        self.fcblock3 = BasicBlock(4 * dim)
        self.fcblock4 = BasicBlock(8 * dim)
        self.fcblock4_1 = BasicBlock(8 * dim)
        self.fcblock4_2 = BasicBlock(8 * dim)
        self.fcblock4_3 = BasicBlock(8 * dim)
        self.fcblock4_4 = BasicBlock(8 * dim)
        self.fcblock4_5 = BasicBlock(8 * dim)
        self.fcblock4_6 = BasicBlock(8 * dim)
        self.fcblock4_7 = BasicBlock(8 * dim)
        self.downsample1 = Downsample(in_chans=dim, out_chans=2 * dim)
        self.downsample2 = Downsample(in_chans=2*dim, out_chans=4 * dim)
        self.downsample3 = Downsample(in_chans=4*dim, out_chans=8 * dim)
        self.upsample1 = Upsample(in_chans=8 * dim, out_chans=4 * dim)
        self.upsample2 = Upsample(in_chans=4 * dim, out_chans=2 * dim)
        self.upsample3 = Upsample(in_chans=2 * dim, out_chans= dim)
        self.conv1 = DynamicFusionBlock_2in(4 * dim)
        self.conv2 = DynamicFusionBlock_2in(2 * dim)
        self.conv3 = DynamicFusionBlock_2in(dim)
        self.fcblock5=BasicBlock(4 * dim)
        self.fcblock6= BasicBlock(2 * dim)
        self.fcblock7= BasicBlock(dim)

    def forward(self, x):
        #encoder
        fcb1=self.fcblock1(x)
        down1=self.downsample1(fcb1)
        fcb2=self.fcblock2(down1)
        down2=self.downsample2(fcb2)
        fcb3 = self.fcblock3(down2)
        down3 = self.downsample3(fcb3)
        fcb4 = self.fcblock4(down3)
        fcb4 = self.fcblock4_1(fcb4)
        fcb4 = self.fcblock4_2(fcb4)
        fcb4 = self.fcblock4_3(fcb4)
        fcb4 = self.fcblock4_4(fcb4)
        fcb4 = self.fcblock4_5(fcb4)
        fcb4 = self.fcblock4_6(fcb4)
        fcb4 = self.fcblock4_7(fcb4)
        #decoder
        up1 = self.upsample1(fcb4)
        up1 = self.conv1(up1, fcb3)
        fcb5 = self.fcblock5(up1)
        up2 = self.upsample2(fcb5)
        up2 = self.conv2(up2, fcb2)
        fcb6 = self.fcblock6(up2)
        up3 = self.upsample3(fcb6)
        up3 = self.conv3(up3, fcb1)
        fcb7 = self.fcblock7(up3)
        return fcb7



class FSNet(nn.Module):
    def __init__(self,dim=32):

        super().__init__()
        self.input_pro = nn.Sequential(
            nn.Conv2d(3,dim,3,1,1),
            nn.GELU(),
            nn.Conv2d(dim,dim,3,1,1)
        )
        self.output_pro=nn.Sequential(
            nn.Conv2d(dim,3,3,1,1)
        )
        self.wave=WavePool(dim)
        self.unwave=WaveUnpool(dim)
        self.convblock_high1=DynamicFusionBlock_3in(dim)
        self.convblock_low1=nn.Sequential(
            nn.Conv2d(dim,dim,1),
            nn.GELU(),
            nn.Conv2d(dim,dim,1)
        )
        self.resunet_low1=ResUNet(dim)
        self.resunet_low2 = ResUNet(dim)
        self.resunet_high1=ResUNet(dim)
        self.resunet_high2 = ResUNet(dim)
        self.coopatt=CooperateAttentionBlock(dim)
        self.convblock_high2 = nn.Sequential(
            nn.Conv2d(dim, 3*dim, 1),
            nn.GELU(),
            nn.Conv2d(3*dim,3*dim,1)
        )
        self.convblock_low2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim,dim,1)
        )

    def forward(self, x):
        #get feature map
        res=x
        x=self.input_pro(x) #c,h,w
        ll, lh, hl, hh=self.wave(x) #c,h/2,w/2
        low=ll
        low=self.convblock_low1(low)
        high=self.convblock_high1(lh,hl,hh)
        high=self.resunet_high1(high)
        low=self.resunet_low1(low)

        high, low=self.coopatt(high,low)
        high=self.resunet_high2(high)
        low=self.resunet_low2(low)
        high=self.convblock_high2(high)
        low=self.convblock_low2(low)
        ll=low
        lh, hl, hh =torch.chunk(high, 3, dim=1)
        y=self.unwave(ll, lh, hl, hh)
        y=self.output_pro(y)
        return  y+res
if __name__ == '__main__':
    input = torch.randn(2, 3,128,128)
    model = FSNet()
    out = model(input)
    # print(model)
    p_number = network_parameters(model)
    p_number = clever_format([p_number], "%.3f")
    print(">>>> model Param.: ", p_number)
    print(out.shape)

