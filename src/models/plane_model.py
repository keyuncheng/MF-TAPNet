import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

class Conv2dReLU(nn.Module):
    """
    [Conv2d(in_channels, out_channels, kernel),
    BatchNorm2d(out_channels),
    ReLU,]
    """
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, bn=False):
        super(Conv2dReLU, self).__init__()
        modules = OrderedDict()
        modules['conv'] = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
        if bn:
            modules['bn'] = nn.BatchNorm2d(out_channels)
        modules['relu'] = nn.ReLU(inplace=True)
        self.l = nn.Sequential(modules)

    def forward(self, x):
        x = self.l(x)
        return x
        

class UNetModule(nn.Module):
    """

    [Conv2dReLU(in_channels, out_channels, 3),
    Conv2dReLU(out_channels, out_channels, 3)]
    """
    def __init__(self, in_channels, out_channels, padding=1, bn=False):
        super(UNetModule, self).__init__()
        self.l = nn.Sequential(OrderedDict([
            ('conv1', Conv2dReLU(in_channels, out_channels, 3, padding=padding, bn=bn)),
            ('conv2', Conv2dReLU(out_channels, out_channels, 3, padding=padding, bn=bn))
            ]))

    def forward(self, x):
        x = self.l(x)
        return x

class Interpolate(nn.Module):
    """
    Wrapper function of interpolate/UpSample Module
    """
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super(Interpolate, self).__init__()
        self.fn = lambda x: nn.functional.interpolate(x, scale_factor=scale_factor,
            mode=mode, align_corners=align_corners)

    def forward(self, x):
        return self.fn(x)
        

class UNet(nn.Module):
    """
    UNet implementation
    pretrained model: None
    Note: this implementation doesn't strictly follow UNet, the kernel sizes are halfed
    """
    def __init__(self, in_channels, num_classes, bn=False):
        super(UNet, self).__init__()

        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample = Interpolate(scale_factor=2,
             mode='bilinear', align_corners=False)
        self.upsample4 = Interpolate(scale_factor=4,
             mode='bilinear', align_corners=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.pool4 = nn.MaxPool2d(4, stride=4)

        self.conv1 = UNetModule(in_channels, 32, bn=bn)
        self.conv2 = UNetModule(32, 64, bn=bn)
        self.conv3 = UNetModule(64, 128, bn=bn)
        self.conv4 = UNetModule(128, 256, bn=bn)
        self.center = UNetModule(256, 512, bn=bn)
        self.up4 = UNetModule(512 + 256, 256)
        self.up3 = UNetModule(256 + 128, 128)
        self.up2 = UNetModule(128 + 64, 64)
        self.up1 = UNetModule(64 + 32, 32)
        # final layer are logits
        self.final = nn.Conv2d(32, num_classes, 1)
        
    def forward(self, x, **kwargs):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        center = self.center(self.pool4(conv4))

        up4 = self.up4(torch.cat([conv4, self.upsample4(center)], 1))
        up3 = self.up3(torch.cat([conv3, self.upsample(up4)], 1))
        up2 = self.up2(torch.cat([conv2, self.upsample(up3)], 1))
        up1 = self.up1(torch.cat([conv1, self.upsample(up2)], 1))

        output = self.final(up1)
        return output


class DecoderModule(nn.Module):
    """
    DecoderModule for UNet11, UNet16
    
    Upsample version:
    [Interpolate(scale_factor, 'bilinear'),
    Con2dReLU(in_channels, mid_channels),
    Conv2dReLU(mid_channels, out_channels),
    ]

    DeConv version:
    [Con2dReLU(in_channels, mid_channels),
    ConvTranspose2d(mid_channels, out_channels, kernel=4, stride=2, pad=1),
    ReLU
    ]
    """
    def __init__(self, in_channels, mid_channels, out_channels, upsample=True):
        super(DecoderModule, self).__init__()
        if upsample:
            modules = OrderedDict([
                ('interpolate', Interpolate(scale_factor=2, mode='bilinear', 
                    align_corners=False)),
                ('conv1', Conv2dReLU(in_channels, mid_channels)),
                ('conv2', Conv2dReLU(mid_channels, out_channels))
                ])
        else:
            modules = OrderedDict([
                ('conv', Conv2dReLU(in_channels, mid_channels)),
                ('deconv', nn.ConvTranspose2d(mid_channels, 
                    out_channels, kernel_size=4, stride=2, padding=1)),
                ('relu', nn.ReLU(inplace=True))
                ])
        self.l = nn.Sequential(modules)

    def forward(self, x):
        return self.l(x)


class UNet11(nn.Module):
    """
    UNet11: use VGG11 as encoder and corresponding DecoderModule as decoder
    pretrained-model: ImageNet
    """
    def __init__(self, in_channels, num_classes, pretrained=True, bn=False, upsample=False):
        super(UNet11, self).__init__()
        if bn:
            self.vgg11 = models.vgg11_bn(pretrained=pretrained).features
            pool_idxs = [3, 7, 14, 21, 28]
        else:
            self.vgg11 = models.vgg11(pretrained=pretrained).features
            pool_idxs = [2, 5, 10, 15, 20]

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = self.vgg11[0:pool_idxs[0]]
        self.conv2 = self.vgg11[pool_idxs[0]+1:pool_idxs[1]]
        self.conv3 = self.vgg11[pool_idxs[1]+1:pool_idxs[2]]
        self.conv4 = self.vgg11[pool_idxs[2]+1:pool_idxs[3]]
        self.conv5 = self.vgg11[pool_idxs[3]+1:pool_idxs[4]]

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
        conv2 = self.conv2(self.pool(conv1)) # 128
        conv3 = self.conv3(self.pool(conv2)) # 256
        conv4 = self.conv4(self.pool(conv3)) # 512
        conv5 = self.conv5(self.pool(conv4)) # 512
        center = self.center(self.pool(conv5)) # 256

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        output = self.final(dec1)
        return output


class UNet16(nn.Module):
    """
    UNet11: use VGG11 as encoder and corresponding DecoderModule as decoder
    pretrained-model: ImageNet
    """
    def __init__(self, in_channels, num_classes, pretrained=True, bn=False, upsample=False):
        super(UNet16, self).__init__()
        
        if bn:
            self.vgg16 = models.vgg16_bn(pretrained=pretrained).features
            pool_idxs = [6, 13, 23, 33, 43]
        else:
            self.vgg16 = models.vgg16(pretrained=pretrained).features
            pool_idxs = [4, 9, 16, 23, 30]

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = self.vgg16[0:pool_idxs[0]]
        self.conv2 = self.vgg16[pool_idxs[0]+1:pool_idxs[1]]
        self.conv3 = self.vgg16[pool_idxs[1]+1:pool_idxs[2]]
        self.conv4 = self.vgg16[pool_idxs[2]+1:pool_idxs[3]]
        self.conv5 = self.vgg16[pool_idxs[3]+1:pool_idxs[4]]

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
        conv2 = self.conv2(self.pool(conv1)) # 128
        conv3 = self.conv3(self.pool(conv2)) # 256
        conv4 = self.conv4(self.pool(conv3)) # 512
        conv5 = self.conv5(self.pool(conv4)) # 512
        center = self.center(self.pool(conv5)) # 256

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        output = self.final(dec1)
        return output

class UNet34(nn.Module):
    """
    UNet34: use ResNet34 as encoder and corresponding DecoderModule as decoder
    pretrained-model: ImageNet
    """
    def __init__(self, in_channels, num_classes, pretrained=True, upsample=False):
        super(UNet34, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = models.resnet34(pretrained=pretrained)
        # resnet already with bn layers
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

class DecoderBlockLinkNet(nn.Module):
    """
    LinkNet DecoderBlock
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.l = nn.Sequential(OrderedDict([
            # Quarter half #{kernels}: B, C, H, W -> B, C/4, H, W
            ('conv1', nn.Conv2d(in_channels, in_channels // 4, 1)),
            ('norm1', nn.BatchNorm2d(in_channels // 4)),
            ('relu1', nn.ReLU(inplace=True)),
            # Upsample: B, C/4, H, W -> B, C/4, 2 * H, 2 * W
            ('deconv2', nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)),
            ('norm2', nn.BatchNorm2d(in_channels // 4)),
            ('relu2', nn.ReLU(inplace=True)),
            # back to original #{kernels}: B, C/4, H, W -> B, C, H, W
            ('conv3', nn.Conv2d(in_channels // 4, out_channels, 1)),
            ('norm3', nn.BatchNorm2d(out_channels)),
            ('relu3', nn.ReLU(inplace=True))
            ]))

    def forward(self, x):
        x = self.l(x)
        return x

class LinkNet34(nn.Module):
    """
    LinkNet34: use ResNet34 as encoder and corresponding DecoderBlockLinkNet as decoder
    pretrained-model: ImageNet
    """
    def __init__(self, in_channels, num_classes, pretrained=True):
        super(LinkNet34, self).__init__()
        resnet34 = models.resnet34(pretrained=pretrained)
        # resnet already with bn layers
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