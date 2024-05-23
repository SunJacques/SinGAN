import PIL.Image
import torch
from PIL import Image
import math

def create_reals(opt):
    reals = []
    real = Image.open(opt.real)
    
    for i in range(0, opt.stop_scale + 1, 1):
        scale = math.pow(opt.scale_factor,opt.stop_scale - i)
        curr_real = imresize(real,scale,opt)
        reals.append(curr_real)
    return reals

def imresize(img,scale_factor,opt):
    real = Image.open("img.jpeg")
    if scale_factor < 1:
        real = real.resize((int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)), Image.ANTIALIAS)
    else:
        real = real.resize((int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)))
    return real