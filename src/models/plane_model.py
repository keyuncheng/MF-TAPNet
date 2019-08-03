import torch
import torch.nn as nn

import torchvision.models as models


class Conv2dReLU(nn.Module):
    """docstring for Conv2dReLU"""
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, bn=False):
        super(Conv2dReLU, self).__init__()
        if bn:
            self.l = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
                )
        else:
            self.l = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        x = self.l(x)
        return x
        

class UNetModule(nn.Module):
    """docstring for UNetModule"""
    def __init__(self, in_channels, out_channels, padding=1, bn=False):
        super(UNetModule, self).__init__()
        self.l1 = Conv2dReLU(in_channels, out_channels, 3, padding=padding, bn=bn)
        self.l2 = Conv2dReLU(out_channels, out_channels, 3, padding=padding, bn=bn)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

class Interpolate(nn.Module):
    """docstring for Interpolate"""
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor,\
            mode=self.mode, align_corners=self.align_corners)
        

# UNet implementation
class UNet(nn.Module):
    """docstring for UNet"""
    def __init__(self, in_channels, num_classes, bn=False):
        super(UNet, self).__init__()
        # not strictly follow UNet, the kernel size are halfed
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.maxpool4 = nn.MaxPool2d(4, stride=4)
        self.conv1 = UNetModule(in_channels, 32, bn=bn)
        self.conv2 = UNetModule(32, 64, bn=bn)
        self.conv3 = UNetModule(64, 128, bn=bn)
        self.conv4 = UNetModule(128, 256, bn=bn)
        self.center = UNetModule(256, 512)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample = Interpolate(scale_factor=2,\
             mode='bilinear', align_corners=False)
        self.upsample4 = Interpolate(scale_factor=4,\
             mode='bilinear', align_corners=False)
        self.up4 = UNetModule(512 + 256, 256)
        self.up3 = UNetModule(256 + 128, 128)
        self.up2 = UNetModule(128 + 64, 64)
        self.up1 = UNetModule(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)
        
    def forward(self, x, **kwargs):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.maxpool(conv1))
        conv3 = self.conv3(self.maxpool(conv2))
        conv4 = self.conv4(self.maxpool(conv3))
        center = self.center(self.maxpool4(conv4))

        up4 = self.up4(torch.cat([conv4, self.upsample4(center)], 1))
        up3 = self.up3(torch.cat([conv3, self.upsample(up4)], 1))
        up2 = self.up2(torch.cat([conv2, self.upsample(up3)], 1))
        up1 = self.up1(torch.cat([conv1, self.upsample(up2)], 1))

        output = self.final(up1)
        return output


class DecoderModule(nn.Module):
    """docstring for DecoderModule"""
    def __init__(self, in_channels, mid_channels, out_channels, upsample=True):
        super(DecoderModule, self).__init__()
        if upsample:
            self.module = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear', align_corners=False),
                # nn.Upsample(scale_factor=2, mode='bilinear'),
                Conv2dReLU(in_channels, mid_channels),
                Conv2dReLU(mid_channels, out_channels)
                )
        else:
            self.module = nn.Sequential(
                Conv2dReLU(in_channels, mid_channels),
                nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=4, \
                    stride=2, padding=1),
                nn.ReLU(inplace=True)
                )
    def forward(self, x):
        return self.module(x)
        

class UNet11(nn.Module):
    """
    docstring for UNet11
    use VGG11 as encoder
    """
    def __init__(self, in_channels, num_classes, pretrained=True, upsample=False):
        super(UNet11, self).__init__()
        self.vgg11 = models.vgg11(pretrained=pretrained).features
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv1 = self.vgg11[0:2]
        self.conv2 = self.vgg11[3:5]
        self.conv3 = self.vgg11[6:10]
        self.conv4 = self.vgg11[11:15]
        self.conv5 = self.vgg11[16:20]

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)



    def forward(self, x, **kwargs):
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.maxpool(conv1)) # 128
        conv3 = self.conv3(self.maxpool(conv2)) # 256
        conv4 = self.conv4(self.maxpool(conv3)) # 512
        conv5 = self.conv5(self.maxpool(conv4)) # 512
        center = self.center(self.maxpool(conv5)) # 256

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        output = self.final(dec1)
        return output


class UNet16(nn.Module):
    """
    docstring for UNet16
    use VGG16 as encoder
    use upsample instead of Deconv
    """
    def __init__(self, in_channels, num_classes, pretrained=True, upsample=True):
        super(UNet16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=pretrained).features
        self.maxpool = nn.MaxPool2d(2, stride=2)


        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.vgg16[0],
                                   self.relu,
                                   self.vgg16[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.vgg16[5],
                                   self.relu,
                                   self.vgg16[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.vgg16[10],
                                   self.relu,
                                   self.vgg16[12],
                                   self.relu,
                                   self.vgg16[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.vgg16[17],
                                   self.relu,
                                   self.vgg16[19],
                                   self.relu,
                                   self.vgg16[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.vgg16[24],
                                   self.relu,
                                   self.vgg16[26],
                                   self.relu,
                                   self.vgg16[28],
                                   self.relu)


        # self.conv1 = self.vgg16[0:4]
        # self.conv2 = self.vgg16[5:9]
        # self.conv3 = self.vgg16[10:16]
        # self.conv4 = self.vgg16[17:23]
        # self.conv5 = self.vgg16[24:30]

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec3 = DecoderModule(256 + 256, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x, **kwargs):
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.maxpool(conv1)) # 128
        conv3 = self.conv3(self.maxpool(conv2)) # 256
        conv4 = self.conv4(self.maxpool(conv3)) # 512
        conv5 = self.conv5(self.maxpool(conv4)) # 512

        center = self.center(self.maxpool(conv5)) # 256

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        output = self.final(dec1)
        return output


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # Quarter half #{kernels}: B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # Upsample: B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # back to original #{kernels}: B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, 1)
        self.norm3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x


class UNet34_sum(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=True):
        super(UNet34_sum, self).__init__()
        resnet34 = models.resnet34(pretrained=pretrained)

        self.firstconv = nn.Sequential(
            resnet34.conv1,
            resnet34.bn1,
            resnet34.relu,
            resnet34.maxpool
        )
        self.encoder1 = resnet34.layer1
        self.encoder2 = resnet34.layer2
        self.encoder3 = resnet34.layer3
        self.encoder4 = resnet34.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(512, 256)
        self.decoder3 = DecoderBlockLinkNet(256, 128)
        self.decoder2 = DecoderBlockLinkNet(128, 64)
        self.decoder1 = DecoderBlockLinkNet(64, 32)

        # Final Classifier
        self.finalconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=2), # downsample
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True)
        )
        
        # return logits
        self.final = nn.Conv2d(32, num_classes, 2, padding=1)

    

    def forward(self, x, **kwargs):
        # Encoder
        x = self.firstconv(x) # 64
        e1 = self.encoder1(x) # 64
        e2 = self.encoder2(e1) # 128
        e3 = self.encoder3(e2) # 256
        e4 = self.encoder4(e3) # 512

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3 # 256
        d3 = self.decoder3(d4) + e2 # 128
        d2 = self.decoder2(d3) + e1 # 64
        d1 = self.decoder1(d2) # 32

        # Final Classification
        finalconv = self.finalconv(d1)
        x_out = self.final(finalconv)
        return x_out



class UNet34(nn.Module):
    """
    use upsample instead of deconv
    """
    def __init__(self, in_channels, num_classes, pretrained=True, upsample=False):
        super(UNet34, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = models.resnet34(pretrained=pretrained)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.center = DecoderModule(512, 512, 256, upsample=upsample)

        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(256 + 256, 512, 256, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(64 + 64, 128, 128, upsample=upsample)
        self.dec1 = DecoderModule(128, 128, 32, upsample=upsample)
        self.dec0 = Conv2dReLU(32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x, **kwargs):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        x_out = self.final(dec0)

        return x_out