from src.config import get_arguments
from src.train import train
from src.functions import read_image, adjust_scales2image
import torch


if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()
    
    opt.device = torch.device("cpu")
    opt.input_name = 'img/balloons.jpeg'
    opt.mode = 'train'
    
    
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    
    real = read_image(opt)
    adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp)
    