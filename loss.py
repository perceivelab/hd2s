import sys
import torch
from torch import nn

def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)

class KLDLoss1vs1(nn.Module):
    def __init__(self, dev='cpu'):
        super(KLDLoss1vs1, self).__init__()
        self.dev=dev

    def KLD(self, inp, trg):
        assert inp.size(0)==trg.size(0), "Sizes of the distributions doesn't match"
        batch_size=inp.size(0)
        kld_tensor=torch.empty(batch_size)
        for k in range(batch_size):
            i = inp[k]/torch.sum(inp[k])
            t = trg[k]/torch.sum(trg[k])
            eps = sys.float_info.epsilon
            kld_tensor[k]= torch.sum(t*torch.log(eps+torch.div(t,(i+eps))))
        return kld_tensor.to(self.dev)

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)