from src.config import get_arguments
from src.model import train

if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()
    
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp)
    