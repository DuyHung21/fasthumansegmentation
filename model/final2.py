import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.criteria import TrainMode

class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.MaxPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input

class Upsample(nn.Module):
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.up = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.up.append(nn.Upsample(scale_factor=2, mode='bilinear'))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for up in self.up:
            input = up(input)
        return input

class InceptionDownsample(nn.Module):
    def __init__(self, nIn, nOut, kSize=3, stride=2):
        super().__init__()

        d = nOut // 3
        d0 = d + nOut % 3

        self.b10 = CBR(nIn, d, 3, 1)
        self.b11 = CBR(d, d, 3, 1)
        self.b12 = CBR(d, d, 3, stride)

        self.b20 = CBR(nIn, d, 3, 1)
        self.b21 = CBR(d, d, 3, stride)

        self.b30 = []
        for i in range(stride//2):
            self.b30.append(nn.MaxPool2d(3, stride=2, padding=1))

        self.b31 = CBR(nIn, d0, 1, 1)

    def forward(self, x):
        block1 = self.b10(x)
        block1 = self.b11(block1)
        block1 = self.b12(block1)

        block2 = self.b20(x)
        block2 = self.b21(block2)

        block3 = self.b30[0](x)
        for pool in self.b30[1:]:
            block3 = pool(block3)
        block3 = self.b31(block3)

        return torch.cat([block1, block2, block3], 1)

class InceptionDownsample_simple(nn.Module):
    def __init__(self, nIn, nOut, kSize=3, stride=2):
        super().__init__()

        d = nOut - nIn

        self.b10 = CBR(nIn, d, kSize, stride)
        self.b20 = []
        for i in range(stride//2):
            self.b20.append(nn.MaxPool2d(3, stride=2, padding=1))

    def forward(self, x):
        block1 = self.b10(x)
        block3 = self.b20[0](x)

        for pool in self.b20[1:]:
            block3 = pool(block3)

        return torch.cat([block1, block3], 1)

class InitBlock(nn.Module):
    def __init__(self, nIn, nOut, kSize=3, stride=2, imgDown=1):
        super().__init__()

        self.conv = CBR(nIn, nOut - 3, kSize, stride)
        self.pooling = InputProjectionA(imgDown)

    def forward(self, input, img):
        project_img = self.pooling(img)
        output = self.conv(input)

        output = torch.cat([project_img, output], 1)

        return output


class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, dilation=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1)/2) * dilation
        #self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), dilation=dilation, bias=False)
        #self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = F.relu

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        #output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output

class ResnetBottleNeck(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, dilation=1, add=True, nHidden=16):
        super().__init__()

        self.conv1 = CBR(nIn, nHidden, 1, 1)
        self.conv2 = CBR(nHidden, nHidden, 3, 1, dilation)
        self.conv3 = CBR(nHidden, nOut, 3, 1)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = F.relu
        self.add = add
    
    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)

        if self.add:
            output = output + input
            #output = self.bn(output)
            output = self.act(output)

        return output

class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        baseFilters = 32
        nFilters = baseFilters

        self.down1 = InitBlock(3, nFilters, 7, 4, 2)
        nFilters = nFilters
        self.conv11 = ResnetBottleNeck(nFilters, nFilters, 3, 1)
        self.conv12 = ResnetBottleNeck(nFilters, nFilters, 3, 1)

        self.down2 = InceptionDownsample_simple(nFilters, 2 * nFilters, 3, 2)
        nFilters = 2 * nFilters
        self.conv21 = ResnetBottleNeck(nFilters, nFilters, 3, 1)
        self.conv22 = ResnetBottleNeck(nFilters, nFilters, 3, 1)

        self.down3 = InceptionDownsample_simple(nFilters, 2 * nFilters, 3, 2)
        nFilters = 2 * nFilters

        self.conv31 = ResnetBottleNeck(nFilters, nFilters, 3, 1, 1)
        self.conv32 = ResnetBottleNeck(nFilters, nFilters, 3, 1, 2)
        self.conv33 = ResnetBottleNeck(nFilters, nFilters, 3, 1, 4)

        self.up3 = Upsample(1)
        self.upconv3 = ResnetBottleNeck(nFilters, nFilters//2, 3, 1, add=False)
        self.upconv31 = ResnetBottleNeck(nFilters, nFilters//2, 3, 1, add=False)
        nFilters = nFilters//2

        self.up2 = Upsample(1)
        self.upconv2 = ResnetBottleNeck(nFilters, nFilters//2, 3, 1, add=False)
        self.upconv21 = ResnetBottleNeck(nFilters, nFilters//2, 3, 1, add=False)

        nFilters = nFilters//2
        self.act = CBR(nFilters, 2, 3, 1)

        self.up1 = Upsample(2)
    
    def forward(self, x):
        d10 = self.down1(x, x)
        d11 = self.conv11(d10)
        d12 = self.conv12(d11)

        d20 = self.down2(d12)    
        d21 = self.conv21(d20)
        d22 = self.conv22(d21)

        d30 = self.down3(d22)    
        d31 = self.conv31(d30)
        d32 = self.conv32(d31)
        d33 = self.conv33(d32)

        up30 = self.up3(d33)
        up31 = self.upconv3(up30)
        up32 = torch.cat([up31, d22], 1)
        up33 = self.upconv31(up32)

        up20 = self.up2(up33)
        up21 = self.upconv2(up20)
        up22 = torch.cat([up21, d12], 1)
        up23 = self.upconv21(up22)

        output = self.act(up23)
        output = self.up1(output)

        return output

class MattingNet(nn.Module):
    '''
        This class define the matting network which lies on top of the ESP Network
    '''

    def __init__(self):
        super().__init__()

        self.convF1 = nn.Conv2d(in_channels=11, out_channels=8, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn = nn.BatchNorm2d(num_features=8)
        self.ReLU = nn.ReLU(inplace=True)
        self.convF2 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # init weights
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, seg):
        seg_softmax = F.softmax(seg, dim=1)
        bg, fg = torch.split(seg_softmax, 1, dim=1)

        # input of feathering layer
        imgSqr = x * x
        imgMasked = x * torch.cat((fg, fg, fg), 1)

        convIn = torch.cat((x, seg_softmax, imgSqr, imgMasked), 1)
        newConvF1 = self.ReLU(self.bn(self.convF1(convIn)))
        newConvF2 = self.convF2(newConvF1)

        a, b, c = torch.split(newConvF2, 1, dim=1)
        alpha = a * fg + b * bg + c
        output = self.sigmoid(alpha)

        return output


class CustomNet(nn.Module):
    def __init__(self, verbose=False):
        super().__init__()

        self.seg = EncoderDecoder()
        self.matt = MattingNet()
        self.mode = TrainMode.SEG
        self.verbose = verbose

    def forward(self, x):
        seg = None
        matt = None
        time_seg = 0
        time_matt = 0

        if self.mode == TrainMode.SEG:
            begin = time.time()
            seg = self.seg(x)
            time_seg = time.time() - begin
        elif self.mode == TrainMode.REFINE:
            begin = time.time()
            seg = self.seg(x)
            time_seg = time.time() - begin

            begin = time.time()
            matt = self.matt(x, seg)
            time_matt = time.time() - begin
        else:
            begin = time.time()
            seg = self.seg(x)
            time_seg = time.time() - begin

            begin = time.time()
            matt = self.matt(x, seg)
            time_matt = time.time() - begin

        if self.verbose:
            print('Time seg: ', time_seg)
            print('Time matt: ', time_matt)
            print('Time overall: ', time_seg + time_matt)
            print()

            return seg, matt, (time_seg, time_seg + time_matt)

        else:
            return seg, matt

    def set_mode(self, mode):
        self.mode = mode

        if mode == TrainMode.SEG:
            for param in self.seg.parameters():
                param.requires_grad = True
            for param in self.matt.parameters():
                param.requires_grad = False

        elif mode == TrainMode.REFINE:
            for param in self.seg.parameters():
                param.requires_grad = False
            for param in self.matt.parameters():
                param.requires_grad = True
        else:
            for param in self.seg.parameters():
                param.requires_grad = True
            for param in self.matt.parameters():
                param.requires_grad = True
