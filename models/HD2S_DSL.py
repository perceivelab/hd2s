import torch
import os
import torch.nn as nn
from itertools import product
from models.S3D_featureExtractor import S3D_featureExtractor_multi_output, BasicConv3d
from models.Decoders_DSL import Decoder2_DS, Decoder3_DS, Decoder4_DS, Decoder5_DS, DomainBatchNorm2d
from models.GaussianFilter.LearnableGaussianFilter import GaussianLayer

__all__ = ['HD2S_DSL']

class HD2S_DSL(nn.Module):
    def __init__(self, pretrained=False, n_gaussians=16, 
                 sources=('DHF1K', 'Hollywood', 'UCFSports'), 
                 domSpec_bn = True, gaussian_priors= True,
                 gaussian_layer = True, max_sigma = 11):
        super(HD2S_DSL, self).__init__()
        
        if isinstance(domSpec_bn, bool):
            self.list_domSpec_bn = [domSpec_bn]*4
        elif isinstance(domSpec_bn, list) and len(domSpec_bn)==4 and all(isinstance(b, bool) for b in domSpec_bn):
            self.list_domSpec_bn = domSpec_bn
        else:
            raise 'Invalid domSpec_bn param'
        
        if isinstance(gaussian_priors, bool):
            self.list_gaussian_priors = [gaussian_priors]*4
        elif isinstance(gaussian_priors, list) and len(gaussian_priors)==4 and all(isinstance(b, bool) for b in gaussian_priors):
            self.list_gaussian_priors = gaussian_priors
        else:
            raise 'Invalid gaussian_priors param'
        
        if isinstance(gaussian_layer, bool):
            self.gaussian_layer = gaussian_layer
        else:
            raise 'Invalid gaussian_layer param'
        
        self.sources=sources
        
        self.featureExtractor=S3D_featureExtractor_multi_output()
        
        #conv_t T/8-->1
        self.conv_t5 = BasicConv3d(1024, 1024, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        in_channel = 1024
        if self.list_gaussian_priors[3]:
            in_channel += n_gaussians
        self.decoder5=Decoder5_DS(in_channel, out_sigmoid=True, domSpec_bn=self.list_domSpec_bn[3], sources=sources)
        
         #conv_t T/8-->1
        self.conv_t4 = BasicConv3d(832, 832, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        in_channel = 832
        if self.list_gaussian_priors[2]:
            in_channel += n_gaussians
        self.decoder4=Decoder4_DS(in_channel, out_sigmoid=True, domSpec_bn=self.list_domSpec_bn[2], sources=sources)
        
         #conv_t T/4-->1
        self.conv_t3_1 = BasicConv3d(480, 480, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t3_2 = BasicConv3d(480, 480, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        in_channel = 480
        if self.list_gaussian_priors[1]:
            in_channel += n_gaussians
        self.decoder3=Decoder3_DS(in_channel, out_sigmoid=True, domSpec_bn=self.list_domSpec_bn[1], sources=sources)
        
        #conv_t T/2-->1
        self.conv_t2_1 = BasicConv3d(192, 192, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t2_2 = BasicConv3d(192, 192, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t2_3 = BasicConv3d(192, 192, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        in_channel = 192
        if self.list_gaussian_priors[0]:
            in_channel += n_gaussians
        self.decoder2=Decoder2_DS(in_channel, out_sigmoid=True, domSpec_bn=self.list_domSpec_bn[0], sources=sources)
        
        # Initialize domain-specific modules
        for source_str in self.sources:
            source_str = f'_{source_str}'.lower()

            # Initialize learned Gaussian priors parameters
            if n_gaussians > 0:
                self.set_gaussians(source_str, n_gaussians)
            
            #Initialize last aggregation block domain-specific (conv+learnableGaussianLayer)
            if self.gaussian_layer:
                self.__setattr__(f'last_block_{source_str}', 
                                 nn.Sequential(*[nn.Conv2d(4, 1, kernel_size=1, stride=1),
                                                 nn.Sigmoid()]))  
                self.__setattr__(f'GL_{source_str}', GaussianLayer(max_sigma=max_sigma))
            else:
                self.__setattr__('last_block', 
                                 nn.Sequential(*[nn.Conv2d(4, 1, kernel_size=1, stride=1),
                                                 nn.Sigmoid()]))  
                
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
            
    @property
    def this_source(self):
        """Return current source for domain-specific BatchNorm."""
        return self._this_source

    @this_source.setter
    def this_source(self, source):
        """Set current source for domain-specific BatchNorm."""
        for module in self.modules():
            if isinstance(module, DomainBatchNorm2d):
                module.this_source = source
        self._this_source = source


    def set_gaussians(self, source_str, n_gaussians):
        """Set Gaussian parameters."""
        suffix = source_str
        for i in range(len(self.list_gaussian_priors)):
            if self.list_gaussian_priors[i]:
                self.__setattr__(
                    f'gaussians{i+2}_' + suffix,
                    self._initialize_gaussians(n_gaussians))

    def _initialize_gaussians(self, n_gaussians, gaussian_init='manual'):
        """
        Return initialized Gaussian parameters.
        Dimensions: [idx, y/x, mu/logstd].
        """
        if gaussian_init == 'manual':
            gaussians = torch.Tensor([
                    list(product([0.25, 0.5, 0.75], repeat=2)) +
                    [(0.5, 0.25), (0.5, 0.5), (0.5, 0.75)] +
                    [(0.25, 0.5), (0.5, 0.5), (0.75, 0.5)] +
                    [(0.5, 0.5)],
                    [(-1.5, -1.5)] * 9 + [(0, -1.5)] * 3 + [(-1.5, 0)] * 3 +
                    [(0, 0)],
            ]).permute(1, 2, 0)

        elif gaussian_init == 'random':
            with torch.no_grad():
                gaussians = torch.stack([
                        torch.randn(
                            n_gaussians, 2, dtype=torch.float) * .1 + 0.5,
                        torch.randn(
                            n_gaussians, 2, dtype=torch.float) * .2 - 1,],
                    dim=2)

        else:
            raise NotImplementedError

        gaussians = nn.Parameter(gaussians, requires_grad=True)
        return gaussians

    @staticmethod
    def _make_gaussian_maps(x, gaussians, size=None, scaling=6.):
        """Construct prior maps from Gaussian parameters."""
        if size is None:
            size = x.shape[-2:]
            bs = x.shape[0]
        else:
            size = [size] * 2
            bs = 1
        dtype = x.dtype
        device = x.device

        gaussian_maps = []
        map_template = torch.ones(*size, dtype=dtype, device=device)
        meshgrids = torch.meshgrid(
            [torch.linspace(0, 1, size[0], dtype=dtype, device=device),
             torch.linspace(0, 1, size[1], dtype=dtype, device=device),])

        for gaussian_idx, yx_mu_logstd in enumerate(torch.unbind(gaussians)):
            map = map_template.clone()
            for mu_logstd, mgrid in zip(yx_mu_logstd, meshgrids):
                mu = mu_logstd[0]
                std = torch.exp(mu_logstd[1])
                map *= torch.exp(-((mgrid - mu) / std) ** 2 / 2)

            map *= scaling
            gaussian_maps.append(map)

        gaussian_maps = torch.stack(gaussian_maps)
        gaussian_maps = gaussian_maps.unsqueeze(0).expand(bs, -1, -1, -1)
        return gaussian_maps

    def _get_gaussian_maps(self, x, source_str, index):
        """Returns the constructed Gaussian prior maps."""
        suffix = source_str
        gaussians = self.__getattr__("gaussians"+ str(index) + "_" + suffix)
        gaussian_maps = self._make_gaussian_maps(x, gaussians)
        return gaussian_maps
                    
    def forward(self, x, source='DHF1K', activate_GL = True, quantization = True):
        
        self.this_source = source
        source_str = f'_{source.lower()}'
        
        features1, features2, features3, features4, features5 = self.featureExtractor(x)
        
        ############# DECONV 5
        x5 = self.conv_t5(features5)
        x5 = x5.squeeze(2)
        
        if self.list_gaussian_priors[3]:
            gaussian_maps = self._get_gaussian_maps(x5, source_str, 5)
            x5=torch.cat((x5,gaussian_maps), dim=1)
        
        sal5=self.decoder5(x5)
        
        ############# DECONV 4
        x4 = self.conv_t4(features4)
        x4 = x4.squeeze(2)
        
        if self.list_gaussian_priors[2]:
            gaussian_maps = self._get_gaussian_maps(x4, source_str, 4)
            x4=torch.cat((x4,gaussian_maps), dim=1)
        
        sal4 = self.decoder4(x4)
        
        ############# DECONV 3
        x3 = self.conv_t3_1(features3)
        x3 = self.conv_t3_2(x3)
        x3 = x3.squeeze(2)
        
        if self.list_gaussian_priors[1]:
            gaussian_maps = self._get_gaussian_maps(x3, source_str, 3)
            x3=torch.cat((x3,gaussian_maps), dim=1)
        
        sal3 = self.decoder3(x3)
        
        ############# DECONV 2
        x2 = self.conv_t2_1(features2)
        x2 = self.conv_t2_2(x2)
        x2 = self.conv_t2_3(x2)
        x2 = x2.squeeze(2)
        
        if self.list_gaussian_priors[0]:
            gaussian_maps = self._get_gaussian_maps(x2, source_str, 2)
            x2=torch.cat((x2,gaussian_maps), dim=1)
        
        sal2 = self.decoder2(x2)
        
        #concatenates saliency map
        out = torch.cat((sal5, sal4, sal3, sal2), 1)
        
        if self.gaussian_layer:
            out = self.__getattr__(f'last_block_{source_str}')(out)
            
            if activate_GL:
                if quantization:
                    dev = out.device
                    out = ((out*255.).type(torch.IntTensor)/255.).to(dev)
                out = self.__getattr__(f'GL_{source_str}')(out)
        else:
            out = self.__getattr__('last_block')(out)
        
        return  sal2.squeeze(1), sal3.squeeze(1), sal4.squeeze(1), sal5.squeeze(1), out.squeeze(1)
    
    

    


