#!/usr/bin/env python

import torch
import math

import layers.sepconv as sepconv

##########################################################

class SepConv(torch.nn.Module):
    def __init__(self):
        super(SepConv, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Upsample(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Subnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
            )
        # end

        self.moduleConv1 = Basic(6, 32)
        self.moduleConv2 = Basic(32, 64)
        self.moduleConv3 = Basic(64, 128)
        self.moduleConv4 = Basic(128, 256)
        self.moduleConv5 = Basic(256, 512)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleDeconv4 = Basic(512, 256)
        self.moduleDeconv3 = Basic(256, 128)
        self.moduleDeconv2 = Basic(128, 64)

        self.moduleUpsample5 = Upsample(512, 512)
        self.moduleUpsample4 = Upsample(256, 256)
        self.moduleUpsample3 = Upsample(128, 128)
        self.moduleUpsample2 = Upsample(64, 64)

        self.moduleVertical1 = Subnet()
        self.moduleVertical2 = Subnet()
        self.moduleHorizontal1 = Subnet()
        self.moduleHorizontal2 = Subnet()

    # end

    def forward(self, tensorFirst, tensorSecond):
        tensorConv1 = self.moduleConv1(torch.cat([ tensorFirst, tensorSecond ], 1))
        tensorConv2 = self.moduleConv2(torch.nn.functional.avg_pool2d(input=tensorConv1, kernel_size=2, stride=2))
        tensorConv3 = self.moduleConv3(torch.nn.functional.avg_pool2d(input=tensorConv2, kernel_size=2, stride=2))
        tensorConv4 = self.moduleConv4(torch.nn.functional.avg_pool2d(input=tensorConv3, kernel_size=2, stride=2))
        tensorConv5 = self.moduleConv5(torch.nn.functional.avg_pool2d(input=tensorConv4, kernel_size=2, stride=2))

        tensorDeconv5 = self.moduleUpsample5(self.moduleDeconv5(torch.nn.functional.avg_pool2d(input=tensorConv5, kernel_size=2, stride=2)))
        tensorDeconv4 = self.moduleUpsample4(self.moduleDeconv4(tensorDeconv5 + tensorConv5))
        tensorDeconv3 = self.moduleUpsample3(self.moduleDeconv3(tensorDeconv4 + tensorConv4))
        tensorDeconv2 = self.moduleUpsample2(self.moduleDeconv2(tensorDeconv3 + tensorConv3))

        tensorCombine = tensorDeconv2 + tensorConv2

        tensorFirst = torch.nn.functional.pad(input=tensorFirst, pad=[ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ], mode='replicate')
        tensorSecond = torch.nn.functional.pad(input=tensorSecond, pad=[ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ], mode='replicate')

        tensorDot1 = sepconv.FunctionSepconv(tensorInput=tensorFirst, tensorVertical=self.moduleVertical1(tensorCombine), tensorHorizontal=self.moduleHorizontal1(tensorCombine))
        tensorDot2 = sepconv.FunctionSepconv(tensorInput=tensorSecond, tensorVertical=self.moduleVertical2(tensorCombine), tensorHorizontal=self.moduleHorizontal2(tensorCombine))

        return tensorDot1 + tensorDot2
    # end
# end
