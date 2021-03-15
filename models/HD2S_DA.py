import torch
import os
import torch.nn as nn
from models.S3D_featureExtractor import S3D_featureExtractor_multi_output, BasicConv3d
from models.Decoders import Decoder2, Decoder3, Decoder4, Decoder5
from models.GRL.functions import ReverseLayerF
from models.GRL.DomainClassifier import DomainClassifier, FeatureReduction

__all__ = ['HD2S_DA']

class HD2S_DA(nn.Module):
    def __init__(self, pretrained=False):
        super(HD2S_DA, self).__init__()
        self.featureExtractor=S3D_featureExtractor_multi_output()
        
        #conv_t T/8-->1
        self.conv_t5 = BasicConv3d(1024, 1024, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.decoder5=Decoder5(1024, out_sigmoid=True)
        
         #conv_t T/8-->1
        self.conv_t4 = BasicConv3d(832, 832, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.decoder4=Decoder4(832, out_sigmoid=True)
        
         #conv_t T/4-->1
        self.conv_t3_1 = BasicConv3d(480, 480, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t3_2 = BasicConv3d(480, 480, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.decoder3=Decoder3(480, out_sigmoid=True)
        
        #conv_t T/2-->1
        self.conv_t2_1 = BasicConv3d(192, 192, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t2_2 = BasicConv3d(192, 192, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t2_3 = BasicConv3d(192, 192, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.decoder2=Decoder2(192, out_sigmoid=True)
        
        self.last_conv = nn.Conv2d(4, 1, kernel_size=1, stride=1)
        
        self.sigmoid = nn.Sigmoid()
        
        #Domain Classifier
        self.featureReduction5=FeatureReduction(bottleneck_channel=1024, ch_out=[512, 64, 16])
        self.domainClassifier5=DomainClassifier(in_features=16*4*6, out_features=[64, 16, 2])
        
        self.featureReduction4=FeatureReduction(bottleneck_channel=832, ch_out=[128, 16])
        self.domainClassifier4=DomainClassifier(in_features=16*4*6, out_features=[64, 16, 2])
        
        self.featureReduction3=FeatureReduction(bottleneck_channel=480, ch_out=[64, 4])
        self.domainClassifier3=DomainClassifier(in_features=4*8*12, out_features=[64, 16, 2])
        
        self.featureReduction2=FeatureReduction(bottleneck_channel=192, ch_out=[64, 32, 4])
        self.domainClassifier2=DomainClassifier(in_features=4*16*24, out_features=[512, 128,32, 2])
        
        
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
                    
    def forward(self, x, alpha=1, out_consp = False):
        
        _, features2, features3, features4, features5 = self.featureExtractor(x)
        
        #DECONV 5
        x5 = self.conv_t5(features5)
        x5 = x5.squeeze(2)
        sal5=self.decoder5(x5)
        
        #DECONV 4
        x4 = self.conv_t4(features4)
        x4 = x4.squeeze(2)
        sal4 = self.decoder4(x4)
        
        #DECONV 3
        x3 = self.conv_t3_1(features3)
        x3 = self.conv_t3_2(x3)
        x3 = x3.squeeze(2)
        sal3 = self.decoder3(x3)
        
        #DECONV 2
        x2 = self.conv_t2_1(features2)
        x2 = self.conv_t2_2(x2)
        x2 = self.conv_t2_3(x2)
        x2 = x2.squeeze(2)
        sal2 = self.decoder2(x2)
        
        x = torch.cat((sal5, sal4, sal3, sal2), 1)
        x = self.last_conv(x)
        x = self.sigmoid(x)
        
        if self.training:
            #GRL+DomainClassifier
            
            y5=ReverseLayerF.apply(x5, alpha)
            features_redux5=self.featureReduction5(y5)
            domain_output5=self.domainClassifier5(features_redux5.view(features_redux5.shape[0],-1))
            
            y4=ReverseLayerF.apply(x4, alpha)
            features_redux4=self.featureReduction4(y4)
            domain_output4=self.domainClassifier4(features_redux4.view(features_redux4.shape[0],-1))
            
            y3=ReverseLayerF.apply(x3, alpha)
            features_redux3=self.featureReduction3(y3)
            domain_output3=self.domainClassifier3(features_redux3.view(features_redux3.shape[0],-1))
            
            y2=ReverseLayerF.apply(x2, alpha)
            features_redux2 = self.featureReduction2(y2)
            domain_output2=self.domainClassifier2(features_redux2.view(features_redux2.shape[0],-1))
            
            return x.squeeze(1), domain_output5, domain_output4, domain_output3, domain_output2
        
        if out_consp:
            return sal2.squeeze(1), sal3.squeeze(1), sal4.squeeze(1), sal5.squeeze(1), x.squeeze(1)
        
        return x.squeeze(1)