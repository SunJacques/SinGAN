import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from src.functions import *
import os

from src.model import *

def train(opt, Gs, Zs, reals, NoiseAmp):
    torch.autograd.set_detect_anomaly(True)
    scale = 0
    reals = create_reals(opt)
    
    while scale < opt.stop_scale + 1:
        opt.out_ = generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale)
        
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass
        
        plt.imsave('%s/real_scale.png' %  (opt.outf), convert_image_np(reals[scale]), vmin=0, vmax=1)
        
        # opt.n_kernel = min(opt.n_kernel_init * pow(2, math.floor(scale / 4)), 128) #double the number of filters every 4 scales
        # opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale / 4)), 128)

        D_curr, G_curr = init_models(opt, scale)
        
        z_curr, G_curr, last_noise_amp = train_single_scale(D_curr, G_curr, reals, Gs, Zs, NoiseAmp, opt)
        
        G_curr = reset_grads(G_curr,False)
        G_curr.eval()
        D_curr = reset_grads(D_curr,False)
        D_curr.eval()
        
        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(last_noise_amp)

def train_single_scale(D, G, reals, Gs, Zs, NoiseAmp, opt):
    real = reals[len(Gs)]
    
    pad = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    m_pad = nn.ZeroPad2d(pad)
    
    z_x = real.shape[2]
    z_y = real.shape[3]
    fixed_noise = generate_noise(3, [z_x, z_y],opt)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    z_opt = m_pad(z_opt)
    
    alpha = opt.alpha
    
    optimizerD = optim.Adam(D.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    
    
    for epoch in range(opt.niter):
        if (Gs == []):
            z_opt = generate_noise(1, [z_x, z_y],opt)
            z_opt = m_pad(z_opt.expand(1,3,z_opt.shape[2],z_opt.shape[3]))
            noise_ = generate_noise(1, [z_x, z_y],opt)
            noise_ = m_pad(noise_.expand(1,3, noise_.shape[2], noise_.shape[3]))
        else:
            noise_ = generate_noise(opt.nc_z, [z_x, z_y],opt)
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
                if (Gs == []):
                    prev = torch.full([1,3,z_x,z_y], 0, device=opt.device)
                    prev = m_pad(prev)
                    z_prev = torch.full([1,3,z_x,z_y], 0, device=opt.device)
                    z_prev = m_pad(z_prev)
                    noise_amp = 1
                else:
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
                
            if (Gs == []):
                noise = noise_
            else:
                noise = noise_amp * noise_ + prev
            
            fake = G(noise.detach(), prev)
            output = D(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            gradient_penalty = calc_gradient_penalty(D, real, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()
            
            
        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
            
        for j in range(opt.Gsteps):
            G.zero_grad()
            output = D(fake)
            errG = -output.mean()
            errG.backward(retain_graph=True)
            if alpha != 0:
                loss = nn.MSELoss()
                Z_opt = noise_amp * z_opt + z_prev
                rec_loss = alpha * loss(G(Z_opt.detach(),z_prev), real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0
            
            optimizerG.step()
            
        if epoch % 25 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))
            
        if epoch % 25 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample.png' %  (opt.outf), convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt).png'    % (opt.outf),  convert_image_np(G(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
    
    return z_opt, G, noise_amp  
            
                
            
def draw_concat(Gs,Zs,reals,NoiseAmp,mode,m_pad,opt):
    real_init = reals[0]
    G_z = torch.full([1,3, real_init.shape[2], real_init.shape[3]], 0, device=opt.device)
    if mode == "rand":
        if len(Gs) > 0:
            for i, (G, Z_opt, noise_amp, real_curr, real_next) in enumerate( zip(Gs, Zs, NoiseAmp, reals, reals[1:])):
                if i == 0:
                    z = generate_noise(1, [Z_opt.shape[2], Z_opt.shape[3]],opt)
                    z = z.expand(1,3,z.shape[2],z.shape[3])
                else:
                    z = generate_noise(3, [Z_opt.shape[2], Z_opt.shape[3]],opt)
                z = m_pad(z)
                G_z = m_pad(G_z)
                z = z * noise_amp + G_z
                G_z = G(z.detach(),G_z)
                G_z = imresize(G_z, 1/opt.scale_factor)
                G_z = G_z[:,:, 0:real_next.shape[2], 0:real_next.shape[3]]
                
    if mode == "rec":
        for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_pad(G_z)
                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
    return G_z
        
            
def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)
            
def generate_noise(channel_z, size,opt):
    noise = torch.randn(1, channel_z, round(size[0]), round(size[1]), device=opt.device)
    m = nn.Upsample(size=[round(size[0]),round(size[1])], mode='bilinear', align_corners=True)
    return m(noise)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model