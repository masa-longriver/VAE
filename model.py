import torch
import torch.nn as nn

def get_outsize(height, width, padding, kernel, stride):
    new_height = ((height + 2 * padding - kernel) / stride) + 1
    new_width  = ((width  + 2 * padding - kernel) / stride) + 1
    
    return int(new_height), int(new_width)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.c_data = config['data']
        self.c_model = config['model']
        self.calc_hiddensize()
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.c_data['channel'],
                      self.c_model['channel1'],
                      kernel_size=self.c_model['kernel'],
                      stride=self.c_model['stride'],
                      padding=self.c_model['padding']),
            nn.BatchNorm2d(self.c_model['channel1']),
            nn.LeakyReLU(self.c_model['neg_slope'], inplace=True),
            nn.Conv2d(self.c_model['channel1'],
                      self.c_model['channel2'],
                      kernel_size=self.c_model['kernel'],
                      stride=self.c_model['stride'],
                      padding=self.c_model['padding']),
            nn.BatchNorm2d(self.c_model['channel2']),
            nn.LeakyReLU(self.c_model['neg_slope'], inplace=True),
            nn.Conv2d(self.c_model['channel2'],
                      self.c_model['channel3'],
                      kernel_size=self.c_model['kernel'],
                      stride=self.c_model['stride'],
                      padding=self.c_model['padding']),
            nn.BatchNorm2d(self.c_model['channel3']),
            nn.LeakyReLU(self.c_model['neg_slope'], inplace=True),
            nn.Conv2d(self.c_model['channel3'],
                      self.c_model['channel4'],
                      kernel_size=self.c_model['kernel'],
                      stride=self.c_model['stride'],
                      padding=self.c_model['padding']),
            nn.BatchNorm2d(self.c_model['channel4']),
            nn.LeakyReLU(self.c_model['neg_slope'], inplace=True),
            nn.Flatten()
        )
        self.linear_mu = nn.Linear(self.c_model['channel4'] * self.out_height * self.out_width,
                                   self.c_model['latent_dim'])
        self.linear_var = nn.Linear(self.c_model['channel4'] * self.out_height * self.out_width,
                                    self.c_model['latent_dim'])
    
    def calc_hiddensize(self):
        out_height1, out_width1 = get_outsize(self.c_data['height'],
                                              self.c_data['width'],
                                              self.c_model['padding'],
                                              self.c_model['kernel'],
                                              self.c_model['stride'])
        out_height2, out_width2 = get_outsize(out_height1,
                                              out_width1,
                                              self.c_model['padding'],
                                              self.c_model['kernel'],
                                              self.c_model['stride'])
        out_height3, out_width3 = get_outsize(out_height2,
                                              out_width2,
                                              self.c_model['padding'],
                                              self.c_model['kernel'],
                                              self.c_model['stride'])
        out_height4, out_width4 = get_outsize(out_height3,
                                              out_width3,
                                              self.c_model['padding'],
                                              self.c_model['kernel'],
                                              self.c_model['stride'])
        self.out_height = out_height4
        self.out_width  = out_width4
        
    def forward(self, x):
        h = self.conv(x)
        mu = self.linear_mu(h)
        logvar = self.linear_var(h)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, config, height, width):
        super(Decoder, self).__init__()
        self.c_data = config['data']
        self.c_model = config['model']
        self.latent_height = height
        self.latent_width  = width

        self.linear = nn.Linear(self.c_model['latent_dim'],
                                self.c_model['channel4'] * height * width)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.c_model['channel4'],
                               self.c_model['channel3'],
                               kernel_size=self.c_model['kernel'],
                               stride=self.c_model['stride'],
                               padding=self.c_model['padding']),
            nn.BatchNorm2d(self.c_model['channel3']),
            nn.LeakyReLU(self.c_model['neg_slope'], inplace=True),
            nn.ConvTranspose2d(self.c_model['channel3'],
                               self.c_model['channel2'],
                               kernel_size=self.c_model['kernel'],
                               stride=self.c_model['stride'],
                               padding=self.c_model['padding']),
            nn.BatchNorm2d(self.c_model['channel2']),
            nn.LeakyReLU(self.c_model['neg_slope'], inplace=True),
            nn.ConvTranspose2d(self.c_model['channel2'],
                               self.c_model['channel1'],
                               kernel_size=self.c_model['kernel'],
                               stride=self.c_model['stride'],
                               padding=self.c_model['padding']),
            nn.BatchNorm2d(self.c_model['channel1']),
            nn.LeakyReLU(self.c_model['neg_slope'], inplace=True),
            nn.ConvTranspose2d(self.c_model['channel1'],
                               self.c_data['channel'],
                               kernel_size=self.c_model['kernel'],
                               stride=self.c_model['stride'],
                               padding=self.c_model['padding']),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        h = self.linear(z)
        h = h.view(h.size(0),
                   self.c_model['channel4'],
                   self.latent_height,
                   self.latent_width)
        x_out = self.deconv(h)

        return x_out
    

class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config,
                               self.encoder.out_height,
                               self.encoder.out_width)
    
    def reparameterize(self, mu, logver):
        std = logver.mul(0.5).exp_()
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_out = self.decoder(z)

        return x_out, mu, logvar

def loss_fn(x_out, x, mu, logvar):
    loss = nn.MSELoss()(x_out, x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return loss + kl