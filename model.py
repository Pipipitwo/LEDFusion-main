
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dataloader
import numpy as np
from torchstat import stat
from dataloader import rgb2ycbcr,ycbcr2rgb
from deconv import DEConv
from coordatt import CoordAtt



class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_att = self.sigmoid(max_out + avg_out)
        x_after_channel = channel_att * x

        # Spatial Attention
        max_out, _ = torch.max(x_after_channel, dim=1, keepdim=True)
        avg_out = torch.mean(x_after_channel, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x_after_spatial = spatial_att * x_after_channel
        return x_after_spatial

class DEConvLeakyRelu2d(nn.Module):
    # DEConv + BatchNorm + Leaky ReLU
    def __init__(self, in_channels):
        super(DEConvLeakyRelu2d, self).__init__()

        self.deconv = DEConv(dim=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.deconv(x)), negative_slope=0.2)

class ConvLeakyRelu2d(nn.Module):
    # convolution + leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1,bias=True):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,stride=stride,bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return F.leaky_relu(self.bn(self.conv1(x)), negative_slope=0.2)

class get_layer1(nn.Module):
    def __init__(self, num_channels, growth):
        super(get_layer1, self).__init__()
        self.conv_1 = ConvLeakyRelu2d(in_channels=num_channels, out_channels=32, kernel_size=3, stride=1, padding=1,bias=True)
        self.conv_2 = ConvLeakyRelu2d(in_channels=32, out_channels=growth, kernel_size=3, stride=1, padding=1,bias=True)
    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = self.conv_2(x1)
        return x1

class denselayer(nn.Module):
    def __init__(self, num_channels, growth):
        super(denselayer, self).__init__()
        self.conv_1 = ConvLeakyRelu2d(in_channels=num_channels, out_channels=growth, kernel_size=3, stride=1, padding=1,bias=True)
        self.sobel = juanji_sobelxy(num_channels)
        self.sobel_conv = nn.Conv2d(num_channels, growth, 1, 1, 0)
    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.sobel(x)
        x2 = self.sobel_conv(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1)



class denselayer_DEConv(nn.Module):
    def __init__(self, num_channels, growth):
        super(denselayer_DEConv, self).__init__()


        self.deconv_block = DEConvLeakyRelu2d(in_channels=num_channels)


        self.conv_1x1 = nn.Conv2d(num_channels, growth, kernel_size=1, stride=1, padding=0, bias=True)

        self.sobel = juanji_sobelxy(num_channels)
        self.sobel_conv = nn.Conv2d(num_channels, growth, 1, 1, 0)

    def forward(self, x):

        x1 = self.deconv_block(x)
        x1 = self.conv_1x1(x1)

        #sobel
        x2 = self.sobel(x)
        x2 = self.sobel_conv(x2)


        return F.leaky_relu(x1 + x2, negative_slope=0.1)

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.num_channels = 2
        self.num_features = 64
        self.growth = 64
        self.conv_layer1 = get_layer1(self.num_channels,self.num_features)
        self.conv_layer2 = denselayer_DEConv(self.num_features,self.growth)
        self.conv_layer3 = denselayer_DEConv(self.num_features*2,self.growth)
        self.conv_layer4 = denselayer_DEConv(self.num_features*3,self.growth)
    def forward(self, x):
        x_max = torch.max(x, dim=1, keepdim=True).values
        x = torch.cat([x_max, x],dim=1)
        layer1 = self.conv_layer1(x)#in 2,out 64
        layer2 = self.conv_layer2(layer1)#in 64,out 64
        layer2 = torch.cat([layer2,layer1],dim=1)
        layer3 = torch.cat([layer2,self.conv_layer3(layer2)],dim=1)#in 64,out 64*3
        layer4 = torch.cat([layer3,self.conv_layer4(layer3)],dim=1)#in 64*3,out 64*4
        return layer4

# input=feature_y_f(经过fusion_net后的融合图像),output=y_f


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.Lrelu = nn.LeakyReLU()
        filter_n = 32


        self.de_conv1 = nn.Conv2d(filter_n*16, filter_n * 4, 3, 1, 1, bias=True)

        self.de_conv2 = nn.Conv2d(filter_n * 4, filter_n * 2, 3, 1, 1, bias=True)
        self.de_conv3 = nn.Conv2d(filter_n * 2, filter_n, 3, 1, 1, bias=True)
        self.de_conv4 = nn.Conv2d(filter_n, 1, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(filter_n * 4)
        self.bn2 = nn.BatchNorm2d(filter_n * 2)
        self.bn3 = nn.BatchNorm2d(filter_n)
        self.rgb2ycbcr = dataloader.rgb2ycbcr
        self.ycbcr2rgb = dataloader.ycbcr2rgb

    def forward(self, feature):
        feature = self.Lrelu(self.bn1(self.de_conv1(feature)))
        feature = self.Lrelu(self.bn2(self.de_conv2(feature)))
        feature = self.Lrelu(self.bn3(self.de_conv3(feature)))
        Y_f = torch.tanh(self.de_conv4(feature))
        return Y_f

# input=feature_vi,feature_ir,output=feature_y_f
class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet,self).__init__()
        self.encoder=encoder().cuda()
        #self.decoder=decoder().cuda()

        encoder_output_channels = 256


        self.CBAM1 = CBAMLayer(encoder_output_channels).cuda()
        self.CBAM2 = CBAMLayer(encoder_output_channels).cuda()


        self.CoordAtt1 = CoordAtt(inp=encoder_output_channels, oup=encoder_output_channels).cuda()
        self.CoordAtt2 = CoordAtt(inp=encoder_output_channels, oup=encoder_output_channels).cuda()

        self.decoder = decoder().cuda()
    def forward(self,vi_clahe_y,ir):
        ir_orig = ir
        feature_vi_en = self.encoder(vi_clahe_y)
        feature_ir = self.encoder(ir_orig)

        fused_mult = feature_vi_en * feature_ir

        feature_vi_en_ca=self.CoordAtt1(feature_vi_en)

        fused_add = self.CBAM1(feature_vi_en_ca) + self.CBAM2(feature_ir)


        feature_y_f = torch.cat([fused_mult, fused_add], dim=1)

        Y_f = self.decoder(feature_y_f)


        save_ir = self.decoder(
            torch.cat([feature_ir * feature_ir, self.CBAM1(self.CoordAtt1(feature_ir)) +self.CBAM2((feature_ir))], dim=1))
        save_vi_en = self.decoder(
            torch.cat([feature_vi_en *feature_vi_en,
                       self.CBAM1(self.CoordAtt1(feature_vi_en)) + self.CBAM2((feature_vi_en))], dim=1))

        save_y_f = self.decoder(feature_y_f)

        return save_ir,save_vi_en,save_y_f,Y_f

class juanji_sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(juanji_sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class BN_Conv2d(nn.Module):
    def __init__(self, in_channels):
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels,64,3,1,1,bias=True),
            nn.BatchNorm2d(64))
        self.Lrelu = nn.LeakyReLU()
    def forward(self, x):
        out=self.Lrelu(self.seq(x))
        out=out.cuda()
        return out

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return edge_magnitude


class StructureExtractor(nn.Module):
    def __init__(self):
        super(StructureExtractor, self).__init__()
        self.sobel_filter = Sobel()
        self.inc = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.down1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        self.up = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(16),
                                nn.ReLU(inplace=True))
        self.outc_block = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.shape[1] == 3:
            x_gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        else:
            x_gray = x
        noisy_sobel = self.sobel_filter(x_gray)
        inc_features = self.inc(noisy_sobel)
        d1 = self.down1(inc_features)
        d2 = self.down2(d1)
        u1 = self.up(d2)
        concat = torch.cat([u1, inc_features], dim=1)
        denoised_map = self.outc_block(concat)
        return denoised_map, noisy_sobel

class SFTLayer(nn.Module):
    def __init__(self, feature_channels):
        super(SFTLayer, self).__init__()
        self.scale_conv = nn.Sequential(nn.Conv2d(1, 16, 1), nn.LeakyReLU(0.2, True),
                                        nn.Conv2d(16, feature_channels, 1))
        self.shift_conv = nn.Sequential(nn.Conv2d(1, 16, 1), nn.LeakyReLU(0.2, True),
                                        nn.Conv2d(16, feature_channels, 1))

    def forward(self, feature, structure_map):
        scale = self.scale_conv(structure_map)
        shift = self.shift_conv(structure_map)
        return feature * (scale + 1) + shift


class StructureGuidedLAN(nn.Module):
    def __init__(self):
        super(StructureGuidedLAN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

        # SFT layers for guidance
        self.sft1 = SFTLayer(number_f)
        self.sft2 = SFTLayer(number_f)
        self.sft4 = SFTLayer(number_f)

    def forward(self, x, structure_map):
        x_in = x
        x1 = self.relu(self.e_conv1(x_in))
        x1 = self.sft1(x1, structure_map)  # Guidance at the beginning

        x2 = self.relu(self.e_conv2(x1))
        x2 = self.sft2(x2, structure_map)  # Guidance after second conv

        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x4 = self.sft4(x4, structure_map)  # Guidance on deeper feature

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        # Iterative enhancement with epsilon for stability
        epsilon = 1e-8

        def enhance_step(img, r):
            return img + r * ((torch.pow(img, 2) - img) / (torch.exp(img) + epsilon))

        x = enhance_step(x_in, r1)
        x = enhance_step(x, r2)
        x = enhance_step(x, r3)
        enhance_image_1 = enhance_step(x, r4)
        x = enhance_step(enhance_image_1, r5)
        x = enhance_step(x, r6)
        x = enhance_step(x, r7)
        enhance_image_final = enhance_step(x, r8)

        # Clamp the final output to be in [0, 1] range
        enhance_image_final = torch.clamp(enhance_image_final, 0, 1)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image_1, enhance_image_final, r


