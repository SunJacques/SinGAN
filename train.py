import torch
import torch.nn as nn
import torch.optim as optim
import math

from model import *

def train(opt):
    scale = 0
    
    while scale < opt.scale_max:
        opt.n_kernel = min(opt.n_kernel_init * pow(2, math.floor(scale / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale / 4)), 128)

        D_curr, G_curr = init_models(opt, scale)
        
        z_curr, G_curr = train_single_scale(D_curr, G_curr, z_curr, opt)
        


def train_single_scale(D, G, Ds, Gs, z_curr, opt):
    
    optimizerD = optim.Adam(D.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    
    m_noise = nn.ZeroPad2d(int(pad_noise))
    
    for epoch in range(opt.niter):
        if (Gs == []):
            z_curr = generate_noise([opt.nc_z, opt.min_size, opt.min_size])
            
            
            
def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)
            
def generate_noise(size ,num_samp=1, scale=1):
    noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=opt.device)
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    noise = upsampling(noise, size[1], size[2])

    return noise