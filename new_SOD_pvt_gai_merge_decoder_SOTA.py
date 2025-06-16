import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parameter import Parameter
from backbone.Res2Net_v1b import *
from backbone.pvtv2 import pvt_v2_b2
import numpy as np
from torchvision import models
#EORSSD-----------
#{'MAE': 0.004602, 'Smeasure': 0.943296, 'adpEm': 0.973166, 'meanEm': 0.977308,
# 'maxEm': 0.982274, 'adpFm': 0.854758, 'meanFm': 0.881085, 'maxFm': 0.900818}


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),

        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.sigmoid(self.fc(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class agg_local(nn.Module):
    def __init__(self, inplanes, inplan):
        super(agg_local, self).__init__()

        self.res = nn.Sequential(nn.Conv2d(inplanes * 2, inplan, 3, padding=1),
                                 nn.BatchNorm2d(inplan),
                                 nn.ReLU(True)
                                 )
        self.LE = le(inplanes)
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes * 2, inplanes, kernel_size=1),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True))

    def forward(self, att_edg, layers_size, layers):
        att_edg_layers = F.interpolate(att_edg, size=layers_size, mode="bilinear", align_corners=False)
        layers_ = self.conv1(layers)
        con,a = self.LE(layers_, att_edg_layers)
        return con,a


class agg_gobal(nn.Module):
    def __init__(self, layers_dim, last_layer_dim):  # 320 512
        super(agg_gobal, self).__init__()
        self.num_s = layers_dim

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(last_layer_dim, layers_dim, 1),
            # nn.BatchNorm2d(layers_dim),
            nn.ReLU(True)
        )
        self.sig = nn.Sigmoid()
        self.seatt = SEAttention(layers_dim)
        self.stt = SpatialAttention()

        self.cov1x1 = nn.Sequential(
            nn.Conv2d(layers_dim, layers_dim // 2, 1, bias=False),
            nn.BatchNorm2d(layers_dim // 2),
            nn.ReLU(True)
        )
        self.conv_feat = nn.Conv2d(last_layer_dim, layers_dim, kernel_size=1)
        self.conv_atten = nn.Conv2d(last_layer_dim, layers_dim, kernel_size=1)
        self.out_conv = nn.Sequential(nn.Conv2d(layers_dim*2, layers_dim, kernel_size=3,padding=1),
                                      nn.BatchNorm2d(layers_dim),
                                      nn.ReLU(True),nn.Conv2d(layers_dim,layers_dim//2,1),nn.BatchNorm2d(layers_dim//2),
                                      nn.ReLU(True)
                                      )
        # self.ema=EMA(layers_dim)

        self.cov1x1 = nn.Sequential(
            nn.Conv2d(last_layer_dim, layers_dim, 1, bias=False),
            nn.BatchNorm2d(layers_dim),
            nn.ReLU(True)
        )

    def forward(self, layers, last_layer):
        # n, c, h, w = layers.size()
        a, b, _, _ = last_layer.size()

        # layers=self.ema(layers)

        last_layer = self.gap(last_layer)
        img_lev = self.se(last_layer)
        sim = self.sig(layers.mul((F.interpolate(self.stt(img_lev), size=layers.size()[2:], mode='bilinear'))))
        enc = self.seatt(sim * layers)

        b, c, h, w = last_layer.size()
        b1, c1, h1, w1 = layers.size()

        # layer=self.ema(layers)

        feat = self.conv_feat(last_layer).view(b, self.num_s, -1)  # 320 64

        atten = self.conv_atten(last_layer).view(b, self.num_s, -1)
        atten = F.softmax(atten, dim=-1)

        descriptors = torch.bmm(feat, atten.permute(0, 2, 1))  # 320 320


        # print(feat.shape)
        # print(atten.permute(0, 2, 1).shape)
        # atten_vectors = F.softmax(self.conv_de(last_layer), dim=1)
        # print(descriptors.matmul(atten_vectors.view(b1, self.num_s, -1)).view(b1, -1, h1, w1).shape)
        # output = descriptors.matmul(layers.view(b1, self.num_s, -1)).view(b1, -1, h1, w1)
        output = descriptors.matmul(layers.view(b1, self.num_s, -1)).view(b1, -1, h1, w1)
        output = output + self.cov1x1(F.interpolate(last_layer, size=[h1, w1], mode="bilinear"))

        ot = self.out_conv(torch.cat([output,enc],1))

        # return self.cov1x1(enc)
        return ot


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))

    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


class sobel(nn.Module):
    def __init__(self, in_channels):
        super(sobel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.ban = nn.BatchNorm2d(1)
        self.sobel_x1, self.sobel_y1 = get_sobel(in_channels, 1)

    def forward(self, x):
        y = run_sobel(self.sobel_x1, self.sobel_y1, x)
        y = F.relu(self.bn(y))
        y = self.conv1(y)
        y = x + y
        y = self.conv2(y)
        y = F.relu(self.ban(y))

        return y


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)




class le(nn.Module):
    def __init__(self, dim):
        super(le, self).__init__()
        self.channel_attention = ChannelAttention(dim)
        self.spatial_attention = SpatialAttention()

    def forward(self, layers, edge):
        layer_edge = layers * edge
        SAT = self.spatial_attention(layer_edge)
        layer_SAT = layers * SAT
        layer_SAT = layers + layer_SAT
        CAT = self.channel_attention(layer_SAT)
        out = layers * CAT
        return out,SAT



class merge(nn.Module):
    def __init__(self, in_channel, plane_dim):
        super(merge, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channel, plane_dim, 1)
        self.bn1 = nn.BatchNorm2d(plane_dim)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, f1, f2):
        f = f1.mul(f2)
        f = self.relu(self.bn(self.conv(f)))
        f = self.maxpool(self.upsample(f))
        f1 = f + f1
        f2 = f + f2
        f1 = self.maxpool(self.upsample(f1))
        f2 = self.maxpool(self.upsample(f2))
        f = f1 + f2
        f = self.relu1(self.bn1(self.conv1(f)))
        return f


class SOD(nn.Module):
    def __init__(self):
        super(SOD, self).__init__()


        filters = [64, 64, 128, 256, 512]


        # PVTv2
        self.pvt = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './backbone/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.pvt.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.pvt.load_state_dict(model_dict)

        self.up1 = sobel(64)
        self.up2 = sobel(128)
        self.opt_edg = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True))
        self.down_edg = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=False) )

        self.merge_1 = nn.Sequential(nn.Conv2d(512, 64, 1),
                                     )

        self.agg_local2 = agg_local(int(320 / 2), int(320 / 2))
        self.merge_2 = nn.Sequential(nn.Conv2d(int(320 / 2) * 2, 64, 3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(True))

        self.agg_gobal1 = agg_gobal(320, 512)

        self.agg_local3 = agg_local(int(128 / 2), int(128 / 2))
        self.merge_3 = nn.Sequential(nn.Conv2d(int(128 / 2) * 2, 64, 3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(True))
        self.agg_gobal2 = agg_gobal(128, 512)

        self.agg_local4 = agg_local(int(64 / 2), int(64 / 2))

        self.agg_gobal3 = agg_gobal(64, 512)
        self.merge_4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(True))

        self.outra4 = nn.Conv2d(64, 1, 1)
        self.outra3 = nn.Conv2d(64, 1, 1)
        self.outra2 = nn.Conv2d(64, 1, 1)
        self.outra1 = nn.Conv2d(64, 1, 1)

        self.deconv1 = nn.ConvTranspose2d(512, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        image_shape = x.size()[2:]
        # PVTv2
        # torch.Size([1, 64, 64, 64])
        # torch.Size([1, 128, 32, 32])
        # torch.Size([1, 320, 16, 16])
        # torch.Size([1, 512, 8, 8])
        trans = self.pvt(x)
        layer1, layer2, layer3, layer4 = trans[0], trans[1], trans[2], trans[3]

        # sobel探索边界
        edg1 = self.up1(layer1)  # (1,1,64,64)
        edg2 = self.up2(layer2)  # (1,1,32,32)
        opt1 = self.opt_edg(edg1)
        opt2 = self.opt_edg(F.interpolate(edg2, scale_factor=2, mode="bilinear"))  # (1,1,64,64)

        con_edg1_2 = torch.cat([opt1, opt2], 1)
        con_edg = self.down_edg(con_edg1_2)
        att_edg = F.sigmoid(con_edg)  # (1,1,64,64)

        # 边界输出
        edge_pre = F.interpolate(att_edg, size=image_shape, mode="bilinear")

        # 全局语义感知（最后一层特例，因为本是最强语义层）layer4

        merge1 = self.merge_1(layer4)
        # 局部信息感知 layer3
        layer3_local_agg,a1 = self.agg_local2(att_edg, layer3.size()[2:], layer3)
        # 全局语义感知 layer3
        layer3_global = self.agg_gobal1(layer3, layer4)
        merge2 = self.merge_2(torch.cat([layer3_local_agg, layer3_global], 1))  # (1, 64, 16, 16)
        # 局部信息感知 layer2
        layer2_local_agg,a2 = self.agg_local3(att_edg, layer2.size()[2:], layer2)
        # 全局语义感知 layer2
        layer2_global = self.agg_gobal2(layer2, layer4)
        merge3 = self.merge_3(torch.cat([layer2_local_agg, layer2_global], 1))  # (1, 64, 32, 32)
        # 局部信息感知 layer1
        layer1_local_agg,a3 = self.agg_local4(att_edg, layer1.size()[2:], layer1)
        # 全局语义感知 layer1
        layer1_global = self.agg_gobal3(layer1, layer4)

        merge4 = self.merge_4(torch.cat([layer1_local_agg, layer1_global], 1))  # (1, 64, 64, 64)


        decoder4 = self.relu(self.bn1(self.deconv1(layer4)))
        decoder3 = self.relu(self.bn2(self.deconv2(decoder4 + merge2)))
        decoder2 = self.relu(self.bn3(self.deconv3(decoder3 + merge3)))
        decoder1 = self.relu(self.bn4(self.deconv4(decoder2 + merge4)))

        lateral_map_4 = F.interpolate(self.outra4(decoder4), size=image_shape, mode='bilinear')

        lateral_map_3 = F.interpolate(self.outra3(decoder3), size=image_shape, mode='bilinear')

        lateral_map_2 = F.interpolate(self.outra2(decoder2), size=image_shape, mode='bilinear')
        lateral_map_1 = F.interpolate(self.outra1(decoder1), size=image_shape, mode='bilinear')

        return lateral_map_1, lateral_map_2, lateral_map_3, lateral_map_4, edge_pre


if __name__ == '__main__':
    from thop import profile
    # from get_model_size import *

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    model = SOD()
    input = torch.randn(1, 3, 352, 352)
    output = model(input)
    flops, params = profile(model, inputs=(input,))
    print('flops(G): %.3f' % (flops / 1e+9))
    print('params(M): %.3f' % (params / 1e+6))
    # getModelSize(model)
