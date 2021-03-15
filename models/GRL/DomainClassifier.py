import torch.nn as nn


class DomainClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(DomainClassifier, self).__init__()
        self.domainClassifier=self._make_domain_classifier(in_features=in_features, out_features=out_features)
        
    def forward(self, x):
        y=self.domainClassifier(x)
        return y

    def _make_domain_classifier(self, in_features, out_features):
        layers=[]
        inp=in_features
        for i, out in enumerate(out_features[:-1]):
            linear=nn.Linear(inp, out)
            layers.append(linear)
            bn=nn.BatchNorm1d(out)
            layers.append(bn)
            relu=nn.ReLU()
            layers.append(relu)
            if i!=0:
                drop=nn.Dropout()
                layers.append(drop)
            inp=out
        layers.append(nn.Linear(inp, out_features[-1]))
        layers.append(nn.LogSoftmax(dim=1))
        
        return nn.Sequential(*layers)


class FeatureReduction(nn.Module):
    def __init__(self, bottleneck_channel, ch_out):
        super(FeatureReduction, self).__init__()
        self.feature_reduction=self._make_feature_reduction(bottleneck_channel, ch_out)
        
    def forward(self, x):
        y=self.feature_reduction(x)
        return y
    
    def _make_feature_reduction(self, bottleneck_channel, ch_out):
        ch_in=bottleneck_channel
        layers=[]
        for c in ch_out:
            conv_l=nn.Conv2d(ch_in, c, kernel_size=1)
            layers.append(conv_l)
            bn=nn.BatchNorm2d(c)
            layers.append(bn)
            relu=nn.ReLU()
            layers.append(relu)
            ch_in=c
    
        return nn.Sequential(*layers)