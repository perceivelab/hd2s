import torch
import os
import torch.nn as nn
from models.S3D_featureExtractor import S3D_featureExtractor_multi_output, BasicConv3d
from models.Decoders import Decoder2, Decoder3, Decoder4, Decoder5

__all__ = ['HD2S']

class HD2S(nn.Module):
    def __init__(self, pretrained=False):
        super(HD2S, self).__init__()
        self.featureExtractor=S3D_featureExtractor_multi_output()
        
        #conv_t: T/8-->1
        self.conv_t5 = BasicConv3d(1024, 1024, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.decoder5=Decoder5(1024, out_sigmoid=True)
        
         #conv_t: T/8-->1
        self.conv_t4 = BasicConv3d(832, 832, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.decoder4=Decoder4(832, out_sigmoid=True)
        
         #conv_t: T/4-->1
        self.conv_t3_1 = BasicConv3d(480, 480, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t3_2 = BasicConv3d(480, 480, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.decoder3=Decoder3(480, out_sigmoid=True)
        
        #conv_t: T/2-->1
        self.conv_t2_1 = BasicConv3d(192, 192, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t2_2 = BasicConv3d(192, 192, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t2_3 = BasicConv3d(192, 192, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.decoder2=Decoder2(192, out_sigmoid=True)
        
        self.last_conv = nn.Conv2d(4, 1, kernel_size=1, stride=1)
        
        self.sigmoid = nn.Sigmoid()
        
        
        if pretrained:
            print('Loading weights...')
            weight_dict=torch.load(os.path.join('models','S3D_kinetics400.pt'))

            model_dict=self.featureExtractor.state_dict()
            
            list_weight_dict=list(weight_dict.items())
            list_model_dict=list(model_dict.items())
            
            for i in range(len(list_model_dict)):
                assert list_model_dict[i][1].shape==list_weight_dict[i][1].shape
                model_dict[list_model_dict[i][0]].copy_(weight_dict[list_weight_dict[i][0]])
            
            for i in range(len(list_model_dict)):
                assert torch.all(torch.eq(model_dict[list_model_dict[i][0]],weight_dict[list_weight_dict[i][0]].to('cpu')))
            print('Loading done!')
                    
    def forward(self, x, out_consp = False):
        
        _, features2, features3, features4, features5 = self.featureExtractor(x)
        
        #DECONV 5
        x5 = self.conv_t5(features5)
        sal5=self.decoder5(x5.squeeze(2))
        
        #DECONV 4
        x4 = self.conv_t4(features4)
        sal4 = self.decoder4(x4.squeeze(2))
        
        #DECONV 3
        x3 = self.conv_t3_1(features3)
        x3 = self.conv_t3_2(x3)
        sal3 = self.decoder3(x3.squeeze(2))
        
        #DECONV 2
        x2 = self.conv_t2_1(features2)
        x2 = self.conv_t2_2(x2)
        x2 = self.conv_t2_3(x2)
        sal2 = self.decoder2(x2.squeeze(2))
        
        x = torch.cat((sal5, sal4, sal3, sal2), 1)
        x = self.last_conv(x)
        out = self.sigmoid(x)
        
        if self.training or out_consp:
            return  sal2.squeeze(1), sal3.squeeze(1), sal4.squeeze(1), sal5.squeeze(1), out.squeeze(1)
        
        return out.squeeze(1)