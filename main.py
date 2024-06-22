from src.config import get_arguments
from src.train import train
from src.functions import read_image, adjust_scales2image
import torch


if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()
    
<<<<<<< HEAD
    opt.device = torch.device("cpu")
    opt.input_name = 'img/balloons.jpeg'
=======
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.input_name = '/home/infres/jsun-22/Documents/SinGAN/img/balloons.jpeg'
>>>>>>> c88ed8f859b8d95d0cb723159c26c857f2902c8f
    opt.mode = 'train'
    
    
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    
    real = read_image(opt)
    adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp)
    