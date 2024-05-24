from src.config import get_arguments
from src.train import train
from src.functions import read_image, adjust_scales2image
import torch

if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()
    
    opt.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    opt.input_name = '/home/infres/jsun-22/Documents/SinGAN/img/birds.jpeg'
    
    
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    
    real = read_image(opt)
    adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp)
    