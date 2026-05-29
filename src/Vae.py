import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,4,2,1),
            nn.ReLU(True),
            nn.Conv2d(32,64,4,2,1),
            nn.ReLU(True),
            nn.Conv2d(64,128,4,2,1),
            nn.ReLU(True),
            nn.Conv2d(128,256,4,2,1),
            nn.ReLU(True),
        )

        self.fc = nn.Linear(256*8*8, latent_dim*2)

        self.fc_dec = nn.Linear(latent_dim,256*8*8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,3,4,2,1),
            nn.Sigmoid()
        )



    def encode(self,x):
        h = self.encoder(x)
        h = h.view(x.size(0),-1)
        h = self.fc(h)
        mu = h[:,:self.latent_dim]
        logvar = h[:,self.latent_dim:]
        return mu, logvar

    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self,z):
        h = self.fc_dec(z)
        h = h.view(z.size(0),256,8,8)
        return self.decoder(h)

    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu,logvar)

        recon = self.decode(z)
    

        return recon, mu, logvar