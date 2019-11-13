import torch
import torch.nn as nn
import torchvision.models as models

from models.plane_model import Conv2dReLU, UNetModule, Interpolate, DecoderModule, DecoderBlockLinkNet

class AttentionModule(nn.Module):
    """
    attention module:
    incorporating attention map and features
    """
    def __init__(self, in_channels, out_channels, scale_factor, bn=False):
        super(AttentionModule, self).__init__()
        self.scale_factor = scale_factor
        self.downsample = Interpolate(scale_factor=scale_factor,\
             mode='bilinear', align_corners=False)

        self.firstconv = Conv2dReLU(in_channels, out_channels, bn=bn)
        # self-learnable attention map
        self.learnable_attmap = nn.Sequential(
            Conv2dReLU(out_channels, 1, 1, padding=0, bn=bn),
            nn.Sigmoid()
            )

    
    def forward(self, x, attmap, **kwargs):
        if self.scale_factor != 1:
            attmap = self.downsample(attmap)
        x = self.firstconv(x)
        output = x + (x * attmap)
        attmap_learned = self.learnable_attmap(output)
        return output, attmap_learned



class TAPNet(nn.Module):
    """docstring for TAPNet"""
    def __init__(self, in_channels, num_classes, bn=False):
        super(TAPNet, self).__init__()

        # half the kernel size
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv1 = UNetModule(in_channels, 32, bn=bn)
        self.conv2 = UNetModule(32, 64, bn=bn)
        self.conv3 = UNetModule(64, 128, bn=bn)
        self.conv4 = UNetModule(128, 256, bn=bn)
        self.center = UNetModule(256, 512, bn=bn)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample = Interpolate(scale_factor=2,\
             mode='bilinear', align_corners=False)
        self.up4 = UNetModule(512 + 256, 256)
        self.up3 = UNetModule(256 + 128, 128)
        self.up2 = UNetModule(128 + 64, 64)
        self.up1 = UNetModule(64 + 32, 32)

        self.att4 = AttentionModule(512 + 256, 512 + 256, 1/8, bn=bn)
        self.att3 = AttentionModule(256 + 128, 256 + 128, 1, bn=bn)
        self.att2 = AttentionModule(128 + 64, 128 + 64, 1, bn=bn)
        self.att1 = AttentionModule(64 + 32, 64 + 32, 1, bn=bn)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)

        
    def forward(self, x, attmap, **kwargs):
        conv1 = self.conv1(x) # 32
        conv2 = self.conv2(self.maxpool(conv1)) # 64
        conv3 = self.conv3(self.maxpool(conv2)) # 128
        conv4 = self.conv4(self.maxpool(conv3)) # 256
        center = self.center(self.maxpool(conv4)) # 512
        
        att4, attmap4 = self.att4(torch.cat([conv4, self.upsample(center)], 1), attmap)
        up4 = self.up4(att4)
        # up4 = self.up4(torch.cat([conv4, self.upsample(center)], 1))
        att3, attmap3 = self.att3(torch.cat([conv3, self.upsample(up4)], 1), self.upsample(attmap4))
        up3 = self.up3(att3)
        att2, attmap2 = self.att2(torch.cat([conv2, self.upsample(up3)], 1), self.upsample(attmap3))
        up2 = self.up2(att2)
        att1, attmap1 = self.att1(torch.cat([conv1, self.upsample(up2)], 1), self.upsample(attmap2))
        up1 = self.up1(att1)

        output = self.final(up1)
        # if output attmap, occupy too much space
        return output


class TAPNet11(nn.Module):
    """
    docstring for TAPNet11
    use VGG11 as encoder
    """
    def __init__(self, in_channels, num_classes, pretrained=True, bn=False, upsample=False):
        super(TAPNet11, self).__init__()
        self.vgg11 = models.vgg11(pretrained=pretrained).features
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv1 = self.vgg11[0:2]
        self.conv2 = self.vgg11[3:5]
        self.conv3 = self.vgg11[6:10]
        self.conv4 = self.vgg11[11:15]
        self.conv5 = self.vgg11[16:20]
        self.upsample = Interpolate(scale_factor=2,\
             mode='bilinear', align_corners=False)

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(256 + 512, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(256 + 512, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(128 + 256, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(64 + 128, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(32 + 64, 32)

        self.att5 = AttentionModule(256 + 512, 256 + 512, 1/16, bn=bn)
        self.att4 = AttentionModule(256 + 512, 256 + 512, 1, bn=bn)
        self.att3 = AttentionModule(128 + 256, 128 + 256, 1, bn=bn)
        self.att2 = AttentionModule(64 + 128, 64 + 128, 1, bn=bn)
        self.att1 = AttentionModule(32 + 64, 32 + 64, 1, bn=bn)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, attmap, **kwargs):
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.maxpool(conv1)) # 128
        conv3 = self.conv3(self.maxpool(conv2)) # 256
        conv4 = self.conv4(self.maxpool(conv3)) # 512
        conv5 = self.conv5(self.maxpool(conv4)) # 512
        center = self.center(self.maxpool(conv5)) # 256

        
        att5, attmap5 = self.att5(torch.cat([center, conv5], 1), attmap)
        dec5 = self.dec5(att5)
        # dec5 = self.dec5(torch.cat([center, conv5], 1))
        att4, attmap4 = self.att4(torch.cat([dec5, conv4], 1), self.upsample(attmap5))
        dec4 = self.dec4(att4)
        # dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        att3, attmap3 = self.att3(torch.cat([dec4, conv3], 1), self.upsample(attmap4))
        dec3 = self.dec3(att3)
        att2, attmap2 = self.att2(torch.cat([dec3, conv2], 1), self.upsample(attmap3))
        dec2 = self.dec2(att2)
        att1, attmap1 = self.att1(torch.cat([dec2, conv1], 1), self.upsample(attmap2))
        dec1 = self.dec1(att1)
        output = self.final(dec1)
        return output


class TAPNet16(nn.Module):
    """
    docstring for TAPNet16
    use VGG16 as encoder
    """
    def __init__(self, in_channels, num_classes, pretrained=False, bn=False, upsample=True):
        super(TAPNet16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=pretrained).features
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv1 = self.vgg16[0:4]
        self.conv2 = self.vgg16[5:9]
        self.conv3 = self.vgg16[10:16]
        self.conv4 = self.vgg16[17:23]
        self.conv5 = self.vgg16[24:30]
        self.upsample = Interpolate(scale_factor=2,\
             mode='bilinear', align_corners=False)

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(256 + 512, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(256 + 512, 512, 256, upsample=upsample)
        self.dec3 = DecoderModule(128 + 256, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(64 + 128, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(32 + 64, 32)

        self.att5 = AttentionModule(256 + 512, 256 + 512, 1/16, bn=bn)
        self.att4 = AttentionModule(256 + 512, 256 + 512, 1, bn=bn)
        self.att3 = AttentionModule(256 + 256, 128 + 256, 1, bn=bn)
        self.att2 = AttentionModule(64 + 128, 64 + 128, 1, bn=bn)
        self.att1 = AttentionModule(32 + 64, 32 + 64, 1, bn=bn)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)



    def forward(self, x, attmap, **kwargs):
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.maxpool(conv1)) # 128
        conv3 = self.conv3(self.maxpool(conv2)) # 256
        conv4 = self.conv4(self.maxpool(conv3)) # 512
        conv5 = self.conv5(self.maxpool(conv4)) # 512

        center = self.center(self.maxpool(conv5)) # 256

        att5, attmap5 = self.att5(torch.cat([center, conv5], 1), attmap)
        dec5 = self.dec5(att5)
        att4, attmap4 = self.att4(torch.cat([dec5, conv4], 1), self.upsample(attmap5))
        dec4 = self.dec4(att4)
        att3, attmap3 = self.att3(torch.cat([dec4, conv3], 1), self.upsample(attmap4))
        dec3 = self.dec3(att3)
        att2, attmap2 = self.att2(torch.cat([dec3, conv2], 1), self.upsample(attmap3))
        dec2 = self.dec2(att2)
        att1, attmap1 = self.att1(torch.cat([dec2, conv1], 1), self.upsample(attmap2))
        dec1 = self.dec1(att1)

        output = self.final(dec1)
        return output


class TAPNet34_sum(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=True, bn=False):
        super(TAPNet34_sum, self).__init__()
        encoder = models.resnet34(pretrained=pretrained)
        self.upsample = Interpolate(scale_factor=2,\
             mode='bilinear', align_corners=False)

        self.firstconv = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool
        )
        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(512, 256)
        self.decoder3 = DecoderBlockLinkNet(256, 128)
        self.decoder2 = DecoderBlockLinkNet(128, 64)
        self.decoder1 = DecoderBlockLinkNet(64, 32)

        self.att4 = AttentionModule(256, 256, 1/16, bn=bn)
        self.att3 = AttentionModule(128, 128, 1, bn=bn)
        self.att2 = AttentionModule(64, 64, 1, bn=bn)
        self.att1 = AttentionModule(32, 32, 1, bn=bn)

        # Final Classifier
        self.finalconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=2), # downsample
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True)
        )
        
        # return logits
        self.final = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x, attmap, **kwargs):
        # Encoder
        x = self.firstconv(x) # 64
        e1 = self.encoder1(x) # 64
        e2 = self.encoder2(e1) # 128
        e3 = self.encoder3(e2) # 256
        e4 = self.encoder4(e3) # 512

        # Decoder with Skip Connections
        dec4 = self.decoder4(e4) + e3
        att4, attmap4 = self.att4(dec4, attmap)
        dec3 = self.decoder3(att4) + e2
        att3, attmap3 = self.att3(dec3, self.upsample(attmap4))
        dec2 = self.decoder2(att3) + e1
        att2, attmap2 = self.att2(dec2, self.upsample(attmap3))
        dec1 = self.decoder1(att2)
        att1, attmap1 = self.att1(dec1, self.upsample(attmap2))

        # Final Classification
        finalconv = self.finalconv(att1)
        x_out = self.final(finalconv)
        return x_out


class TAPNet34(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=True, bn=False):
        super(TAPNet34, self).__init__()

        self.encoder = models.resnet34(pretrained=pretrained)
        self.firstconv = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.encoder.maxpool)

        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4
        self.upsample = Interpolate(scale_factor=2,\
             mode='bilinear', align_corners=False)

        self.dec4 = DecoderBlockLinkNet(512, 256)
        self.dec3 = DecoderBlockLinkNet(256, 128)
        self.dec2 = DecoderBlockLinkNet(128, 64)
        self.dec1 = DecoderBlockLinkNet(64, 32)

        self.att4 = AttentionModule(256 + 256, 256, 1/16, bn=bn)
        self.att3 = AttentionModule(128 + 128, 128, 1, bn=bn)
        self.att2 = AttentionModule(64 + 64, 64, 1, bn=bn)
        self.att1 = AttentionModule(32, 32, 1, bn=bn)

        # Final Classifier
        self.finalconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, stride=2), # downsample
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True)
        )

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, attmap, **kwargs):
        firstconv = self.firstconv(x) # 64
        conv1 = self.conv1(firstconv) # 64
        conv2 = self.conv2(conv1) # 128
        conv3 = self.conv3(conv2) # 256
        conv4 = self.conv4(conv3) # 512

        dec4 = self.dec4(conv4)
        att4, attmap4 = self.att4(torch.cat([dec4, conv3], 1), attmap)
        dec3 = self.dec3(att4)
        att3, attmap3 = self.att3(torch.cat([dec3, conv2], 1), self.upsample(attmap4))
        dec2 = self.dec2(att3)
        att2, attmap2 = self.att2(torch.cat([dec2, conv1], 1), self.upsample(attmap3))
        dec1 = self.dec1(att2)

        finalconv = self.finalconv(dec1)
        output = self.final(finalconv)

        return output