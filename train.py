import torch
import torch.nn as nn
import torch.optim as optim
import math
from src.functions import create_reals

from src.model import *

def train(opt, Gs, Zs, reals, NoiseAmp):
    scale = 0
    
    reals = create_reals(opt)
    
    while scale < opt.scale_max:
        opt.n_kernel = min(opt.n_kernel_init * pow(2, math.floor(scale / 4)), 128) #double the number of filters every 4 scales
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale / 4)), 128)
        
        D_curr, G_curr = init_models(opt, scale)
        
        z_curr, G_curr = train_single_scale(D_curr, G_curr, reals, Gs, Zs, NoiseAmp, opt)
        
        G_curr = reset_grads(G_curr,False)
        G_curr.eval()
        D_curr = reset_grads(D_curr,False)
        D_curr.eval()
        
        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

def train_single_scale(D, G, reals, Gs, Zs, NoiseAmp, opt):
    real = reals[len(Gs)]
    
    opt.nzx = real.shape[2]
    opt.nzy = real.shape[3]
    fixed_noise = generate_noise(opt.nc_z, [opt.nzx, opt.nzy])
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    z_opt = m_pad(z_opt)
    
    pad = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    m_pad = nn.ZeroPad2d(pad)
    
    alpha = opt.alpha
    
    optimizerD = optim.Adam(D.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    
    
    for epoch in range(opt.niter):
        if (Gs == []):
            z_opt = generate_noise(opt.nc_z, [opt.nzx, opt.nzy])
            z_opt = m_pad(z_opt)
            noise_ = generate_noise(opt.nc_z, [opt.nzx, opt.nzy])
            noise_ = m_pad(noise_)
        else:
            noise_ = generate_noise(opt.nc_z, [opt.nzx, opt.nzy])
            noise_ = m_pad(noise_)
            
        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
            
        for j in range (opt.Dsteps):
            D.zero_grad()
            
            output = D(real).to(opt.device)
            errD_real = -output.mean()
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()
            
            # train with fake
            
            if j==0 and epoch == 0:
                prev = draw_concat(Gs,Zs,reals,NoiseAmp,'rand',m_pad,opt)
                prev = m_pad(prev)
                z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,'rec',m_pad,opt)
                criterion = nn.MSELoss()
                RMSE = torch.sqrt(criterion(real, z_prev))
                opt.noise_amp = opt.noise_amp_init*RMSE
                z_prev = m_pad(z_prev)
                
            else:
                prev = draw_concat(Gs,Zs,reals,NoiseAmp,'rand',m_pad,opt)
                prev = m_pad(prev)
                
            
def draw_concat(Gs,Zs,reals,NoiseAmp,mode,m_pad,opt):
    
            
            
            
            
def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)
            
def generate_noise(channel_z, size ,num_samp=1, scale=1):
    noise = torch.randn(num_samp, channel_z, round(size[0]/scale), round(size[1]/scale), device=opt.device)
    m = nn.Upsample(size=[round(size[0]),round(size[1])], mode='bilinear', align_corners=True)

    return m(noise)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model