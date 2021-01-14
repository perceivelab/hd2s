import torch.nn as nn
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners=align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
    
    
class _SepConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(_SepConv2d, self).__init__()
        self.conv_s = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=in_planes)
        self.bn_s = nn.BatchNorm2d(out_planes)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_t = nn.BatchNorm2d(out_planes)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x 

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


    
class Decoder5(nn.Module):
    def __init__(self, in_channel=1024, out_channel=[512,256, 128, 64], out_sigmoid=False):
        super(Decoder5, self).__init__()
        
        self.out_sigmoid=out_sigmoid
        
        self.deconvlayer5_5 = self._make_deconv(in_channel, out_channel[0], num_conv=3)
        self.upsample5_5=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer5_4 = self._make_deconv(out_channel[0], out_channel[1], num_conv=3)
        self.upsample5_4=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer5_3 = self._make_deconv(out_channel[1], out_channel[2])
        self.upsample5_3=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer5_2 = self._make_deconv(out_channel[2], out_channel[3])
        self.upsample5_2=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer5_1 = self._make_deconv(out_channel[3], 3)
        self.upsample5_1=Upsample(scale_factor=2, mode='bilinear')
        
        
        self.last_conv5=nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=True)
        
        if self.out_sigmoid:
            self.sigmoid= nn.Sigmoid()
     
    def _make_deconv(self, in_channel, out_channel, num_conv=2, kernel_size=3, stride=1, padding=1):
        layers=[]
        layers.append(BasicConv2d(in_channel, out_channel ,kernel_size=kernel_size, stride=stride, padding=padding))
        for i in range(1, num_conv):
            layers.append(_SepConv2d(out_channel, out_channel,kernel_size=kernel_size, stride=stride, padding=padding))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x=self.deconvlayer5_5(x)
        x = self.upsample5_5(x)
        x=self.deconvlayer5_4(x)
        x = self.upsample5_4(x)
        x=self.deconvlayer5_3(x)
        x = self.upsample5_3(x)
        x=self.deconvlayer5_2(x)
        x = self.upsample5_2(x)
        x=self.deconvlayer5_1(x)
        x = self.upsample5_1(x)
        x = self.last_conv5(x)
        
        if self.out_sigmoid:
            x=self.sigmoid(x)
        
        return x
    
class Decoder4(nn.Module):
    def __init__(self, in_channel=832, out_channel=[512, 256, 128, 64], out_sigmoid=False):
        super(Decoder4, self).__init__()
        
        self.out_sigmoid=out_sigmoid
        
        self.deconvlayer4_5 = self._make_deconv(in_channel, out_channel[0], num_conv=3)
        self.upsample4_5=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer4_4 = self._make_deconv(out_channel[0], out_channel[1], num_conv=3)
        self.upsample4_4=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer4_3 = self._make_deconv(out_channel[1], out_channel[2])
        self.upsample4_3=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer4_2 = self._make_deconv(out_channel[2], out_channel[3])
        self.upsample4_2=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer4_1 = self._make_deconv(out_channel[3], 3)
        self.upsample4_1=Upsample(scale_factor=2, mode='bilinear')
        
        
        self.last_conv4=nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=True)
        
        if self.out_sigmoid:
            self.sigmoid= nn.Sigmoid()
    
    def _make_deconv(self, in_channel, out_channel, num_conv=2, kernel_size=3, stride=1, padding=1):
        layers=[]
        layers.append(BasicConv2d(in_channel, out_channel,kernel_size=kernel_size, stride=stride, padding=padding))
        for i in range(1, num_conv):
            layers.append(_SepConv2d(out_channel, out_channel,kernel_size=kernel_size, stride=stride, padding=padding))
        return nn.Sequential(*layers)
    
    def forward(self, x):
       
        x=self.deconvlayer4_5(x)
        x = self.upsample4_5(x)
        x=self.deconvlayer4_4(x)
        x = self.upsample4_4(x)
        x=self.deconvlayer4_3(x)
        x = self.upsample4_3(x)
        x=self.deconvlayer4_2(x)
        x = self.upsample4_2(x)
        x=self.deconvlayer4_1(x)
        x = self.upsample4_1(x)
        
        x = self.last_conv4(x)
        
        if self.out_sigmoid:
            x=self.sigmoid(x)
        
        return x

class Decoder3(nn.Module):
    def __init__(self, in_channel=480, out_channel=[256,128, 64], out_sigmoid=False):
        super(Decoder3, self).__init__()
        
        self.out_sigmoid=out_sigmoid
        
        self.deconvlayer3_4 = self._make_deconv(in_channel, out_channel[0])
        self.upsample3_4=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer3_3 = self._make_deconv(out_channel[0], out_channel[1])
        self.upsample3_3=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer3_2 = self._make_deconv(out_channel[1], out_channel[2])
        self.upsample3_2=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer3_1 = self._make_deconv(out_channel[2], 3)
        self.upsample3_1=Upsample(scale_factor=2, mode='bilinear')
        
        
        self.last_conv3=nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=True)
        
        if self.out_sigmoid:
            self.sigmoid= nn.Sigmoid()
    
    def _make_deconv(self, in_channel, out_channel, num_conv=2, kernel_size=3, stride=1, padding=1):
        layers=[]
        layers.append(BasicConv2d(in_channel, out_channel,kernel_size=kernel_size, stride=stride, padding=padding))
        for i in range(1, num_conv):
            layers.append(_SepConv2d(out_channel, out_channel,kernel_size=kernel_size, stride=stride, padding=padding))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
       
        x=self.deconvlayer3_4(x)
        x = self.upsample3_4(x)
        x=self.deconvlayer3_3(x)
        x = self.upsample3_3(x)
        x=self.deconvlayer3_2(x)
        x = self.upsample3_2(x)
        x=self.deconvlayer3_1(x)
        x = self.upsample3_1(x)
    
        x = self.last_conv3(x)
        
        if self.out_sigmoid:
            x=self.sigmoid(x)
            
        return x

class Decoder2(nn.Module):
    def __init__(self, in_channel=192, out_channel=[128,64], out_sigmoid=False):
        super(Decoder2, self).__init__()
        
        self.out_sigmoid=out_sigmoid
        
        self.deconvlayer2_3 = self._make_deconv(in_channel, out_channel[0])
        self.upsample2_3=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer2_2 = self._make_deconv(out_channel[0], out_channel[1])
        self.upsample2_2=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer2_1 = self._make_deconv(out_channel[1], 3)
        self.upsample2_1=Upsample(scale_factor=2, mode='bilinear')
        
        self.last_conv2=nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=True)
        
        if self.out_sigmoid:
            self.sigmoid= nn.Sigmoid()
    
    def _make_deconv(self, in_channel, out_channel, num_conv=2, kernel_size=3, stride=1, padding=1):
        layers=[]
        layers.append(BasicConv2d(in_channel, out_channel,kernel_size=kernel_size, stride=stride, padding=padding))
        for i in range(1, num_conv):
            layers.append(_SepConv2d(out_channel, out_channel,kernel_size=kernel_size, stride=stride, padding=padding))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
       
        x=self.deconvlayer2_3(x)
        x = self.upsample2_3(x)
        x=self.deconvlayer2_2(x)
        x = self.upsample2_2(x)
        x=self.deconvlayer2_1(x)
        x = self.upsample2_1(x)
        
        x = self.last_conv2(x)
        
        if self.out_sigmoid:
            x=self.sigmoid(x)
            
        return x

class Decoder1(nn.Module):
    def __init__(self, in_channel=64, out_channel=[32], out_sigmoid=False):
        super(Decoder1, self).__init__()
        
        self.out_sigmoid=out_sigmoid
        
        self.deconvlayer1_2 = self._make_deconv(in_channel, out_channel[0])
        self.upsample1_2=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer1_1 = self._make_deconv(out_channel[0], 3)
        self.upsample1_1=Upsample(scale_factor=2, mode='bilinear')
        
        
        self.last_conv1=nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=True)
        
        if self.out_sigmoid:
            self.sigmoid= nn.Sigmoid()
       
    def _make_deconv(self, in_channel, out_channel, num_conv=2, kernel_size=3, stride=1, padding=1):
        layers=[]
        layers.append(BasicConv2d(in_channel, out_channel,kernel_size=kernel_size, stride=stride, padding=padding))
        for i in range(1, num_conv):
            layers.append(_SepConv2d(out_channel, out_channel,kernel_size=kernel_size, stride=stride, padding=padding))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
       
        x=self.deconvlayer1_2(x)
        x = self.upsample1_2(x)
        x=self.deconvlayer1_1(x)
        x = self.upsample1_1(x)
        
        x = self.last_conv1(x)
        
        if self.out_sigmoid:
            x=self.sigmoid(x)
            
        return x
    
    
