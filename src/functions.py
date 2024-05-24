import PIL.Image
import torch
from PIL import Image
import math
import numpy as np
import torchvision.transforms as T

def create_reals(opt):
    reals = []
    real = read_image(opt)
    
    for i in range(0, opt.stop_scale + 1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale - i)
        curr_real = imresize(real,scale)
        reals.append(curr_real)
    return reals

def imresize(img,scale_factor):
    if scale_factor < 1:
        real = T.Resize((int(img.shape[2]*scale_factor), int(img.shape[3]*scale_factor)), antialias=True)(img)
    else:
        real = T.Resize((int(img.shape[2]*scale_factor), int(img.shape[3]*scale_factor)))(img)
    return real

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1,1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def read_image(opt):
    img = Image.open(opt.input_name)
    img = img.convert('RGB')
    img = T.ToTensor()(img)
    img = img.view(1,img.size(0),img.size(1),img.size(2))
    return img

def torch2numpy(img):
    img = img.squeeze(0).permute(1,2,0).numpy()
    return img

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

def adjust_scales2image(real_, opt):
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor))
    opt.stop_scale = opt.num_scales - scale2stop

def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = inp[-1,:,:,:].to(torch.device('cpu'))
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = inp[-1,-1,:,:].to(torch.device('cpu'))
        inp = inp.numpy().transpose((0,1))

    inp = np.clip(inp,0,1)
    return inp

def generate_dir2save(opt):
    dir2save = 'TrainedModels/%s/scale_factor=%f,alpha=%d' % (opt.input_name.split("/")[-1], opt.scale_factor,opt.alpha)
    return dir2save