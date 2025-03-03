# -*- coding: utf-8 -*-
import torch

from torch import Tensor
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.act = nn.ELU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.InstanceNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        return out

class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k = 3, n_layers = 2, pooling = 'max'):
        super(ResBlock, self).__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for ix in range(n_layers):
            self.convs.append(nn.Conv2d(in_channels, out_channels, (k, k), 
                                        stride = (1, 1), padding = ((k + 1) // 2 - 1, (k + 1) // 2 - 1)))
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(out_channels), nn.Dropout2d(0.1)))
            
            in_channels = out_channels
    
        self.activation = nn.ELU()
        
    def forward(self, x, return_unpooled = False):
        xs = [self.norms[0](self.convs[0](x))]
        
        for ix in range(1, len(self.norms)):
            xs.append(self.norms[ix](self.convs[ix](xs[-1])) + xs[-1])
            
        x = self.activation(torch.cat(xs, dim = 1))
        
        return x

# in V2 I replace the VGG Block with Residual convolution blocks with the ELU activation function
class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, filter_multiplier = 1, 
                 deep_supervision = False, small = False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        nb_filter = list(map(int, [u * filter_multiplier for u in nb_filter]))

        self.deep_supervision = deep_supervision
        self.small = small

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)

        self.dropout = nn.Dropout2d(0.1)

        self.conv0_0 = ResBlock(input_channels, nb_filter[0] // 2)
        self.conv1_0 = ResBlock(nb_filter[0], nb_filter[1] // 2)
        self.conv2_0 = ResBlock(nb_filter[1], nb_filter[2] // 2)
        self.conv3_0 = ResBlock(nb_filter[2], nb_filter[3] // 2)
        
        if not self.small:
            self.conv4_0 = ResBlock(nb_filter[3], nb_filter[4] // 2)
            self.conv0_4 = ResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0] // 2)
            
            self.conv2_2 = ResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2] // 2)
            self.conv1_3 = ResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1] // 2)
            self.conv3_1 = ResBlock(nb_filter[3]+nb_filter[4], nb_filter[3] // 2)

        self.conv0_1 = ResBlock(nb_filter[0]+nb_filter[1], nb_filter[0] // 2)
        self.conv1_1 = ResBlock(nb_filter[1]+nb_filter[2], nb_filter[1] // 2)
        self.conv2_1 = ResBlock(nb_filter[2]+nb_filter[3], nb_filter[2] // 2)

        self.conv0_2 = ResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0] // 2)
        self.conv1_2 = ResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1] // 2)

        self.conv0_3 = ResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0] // 2)
        

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        if not self.small:
            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            if not self.small:
                output = torch.squeeze(self.final(x0_4))
            else:
                output = torch.squeeze(self.final(x0_3))
            return output
