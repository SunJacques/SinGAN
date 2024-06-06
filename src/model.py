import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def init_models(opt, scale):
    netG = Generator(opt, scale).to(opt.device)
    netD = Discriminator(opt, scale).to(opt.device)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    return netD, netG
    
class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

class Generator (nn.Module):
    def __init__(self, opt, scale):
        super(Generator, self).__init__()
        N_init = opt.n_filters * pow(2, scale // 4)
        
        self.head = ConvBlock(in_channel=3, out_channel=N_init, ker_size=3, padd=5, stride=1)
        
        self.body = nn.Sequential()
        for i in range(3):
            N = int(N_init / pow(2, i + 1))
            block = ConvBlock(in_channel=max(2*N, 32), out_channel=max(N, 32), ker_size=3, padd=0, stride=1)
            self.body.add_module('block%d'%(i+1),block)
            
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=max(N,32), out_channels=3, kernel_size=3, stride=1, padding=0),
            nn.Tanh()
        )
        
    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        # ind = int((y.shape[2]-x.shape[2])/2)
        # y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x + y

class Discriminator(nn.Module):
    def __init__(self,opt,scale):
        super(Discriminator,self).__init__()
        N_init = opt.n_filters * pow(2, scale // 4)
        self.head = ConvBlock(in_channel=3, out_channel=N_init, ker_size=3, padd=0, stride=1)
        
        self.body = nn.Sequential()
        for i in range(3):
            N = int(N_init / pow(2, i + 1))
            block = ConvBlock(in_channel=max(2*N,32), out_channel=max(N,32), ker_size=3, padd=0, stride=1)
            self.body.add_module('block%d'%(i+1),block)
            
        self.tail = nn.Conv2d(in_channels=max(N,32), out_channels=1, kernel_size=3, stride=1, padding=0)
        
    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x