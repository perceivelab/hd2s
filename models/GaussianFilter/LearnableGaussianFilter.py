import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class GaussianLayer(nn.Module):
    
    def __init__(self, max_sigma, normalize=True):
        super().__init__()
        self.max_sigma = max_sigma
        self.kernel_size = math.ceil(6*self.max_sigma)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.half_k = self.kernel_size//2
        self.sigma = nn.Parameter(torch.tensor(max_sigma/2))
        self.normalize = normalize
        
    def forward(self, x):
        # Computes kernel values for current sigma
        r = torch.arange(-self.half_k, self.half_k+1)
        grid_x, grid_y = torch.meshgrid(r, r)
        grid_x, grid_y = grid_x.float(), grid_y.float()
        
        grid_x= grid_x.to(x.device)
        grid_y= grid_y.to(x.device)
        
        kernel = torch.exp(-(grid_x**2 + grid_y**2)/(2*self.sigma*self.sigma))
        if self.normalize:
            kernel = kernel/kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        x = F.conv2d(x, kernel, padding=self.half_k)
        return x



