from src.config import get_arguments
from src.functions import *
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()
    
    opt.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    opt.input_name = '/home/infres/jsun-22/Documents/SinGAN/img/balloons.jpeg'
    opt.mode = 'random_samples'
    
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = generate_dir2save(opt)
    num_samples = 20
    stop_scale = 8
    
    real = read_image(opt)
    adjust_scales2image(real, opt)
        
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    
    Gs, Zs, reals, NoiseAmp = load_trained_pyramid(opt, scale = stop_scale)
    
    for img_num in range (num_samples):
        real_init = reals[0]
        I_prev = torch.full([1,3, real_init.shape[2], real_init.shape[3]], 0, device=opt.device)
        
        for i, (G, Z_opt, noise_amp, real_next) in enumerate(zip(Gs, Zs, NoiseAmp, reals[1:])):
            G.eval()
            nzx = Z_opt.shape[2] 
            nzy = Z_opt.shape[3] 

            if i == 0:
                z_curr = generate_noise(1,[nzx,nzy], opt)
                z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
            else:
                z_curr = generate_noise(1, [nzx,nzy],opt)

            z_in = noise_amp * (z_curr) + I_prev
            I_curr = G(z_in.detach(), I_prev)
            if i != stop_scale:
                I_prev = upsampling(I_curr, real_next.shape[2], real_next.shape[3])
            
        plt.imsave('%s/fake%d.png'    % (dir2save,img_num),  convert_image_np(I_curr), vmin=0, vmax=1)
    
    