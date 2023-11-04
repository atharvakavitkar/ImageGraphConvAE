#import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

with open("./config.yaml", 'r') as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        a=10
        #print(exc)


#model class

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = params['num_channels']
        self.conv11 = nn.Conv2d(in_channels=3, out_channels=c, kernel_size=2, stride=1, padding=0) 
        self.conv12 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=2, stride=1, padding=0)

        self.conv21 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=2, padding=0)
        self.conv22 = nn.Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=2, stride=1, padding=0)

        self.conv31 = nn.Conv2d(in_channels=c*2, out_channels=c*2*2, kernel_size=3, stride=2, padding=0)
        self.conv32 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=2, stride=1, padding=0)

        self.conv41 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2*2, kernel_size=3, stride=2, padding=0)
        self.conv42 = nn.Conv2d(in_channels=c*2*2*2, out_channels=c*2*2*2, kernel_size=2, stride=1, padding=0)

        self.conv51 = nn.Conv2d(in_channels=c*2*2*2, out_channels=c*2*2*2*2, kernel_size=3, stride=2, padding=0)
        self.conv52 = nn.Conv2d(in_channels=c*2*2*2*2, out_channels=c*2*2*2*2, kernel_size=2, stride=1, padding=0)

        self.conv61 = nn.Conv2d(in_channels=c*2*2*2*2, out_channels=c*2*2*2*2, kernel_size=3, stride=2, padding=0)

        self.fc_mu = nn.Linear(in_features=9216, out_features=params['latent_dims'])
        self.fc_logvar = nn.Linear(in_features=9216, out_features=params['latent_dims'])
            
    def forward(self, x):
        #print("Encoder:")
        #print("Input:",x.shape)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv12(x))
        #print("After conv1:",x.shape)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv22(x))
        #print("After conv2:",x.shape)
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv32(x))
        #print("After conv3:",x.shape)
        x = F.relu(self.conv41(x))
        x = F.relu(self.conv42(x))
        x = F.relu(self.conv42(x))
        #print("After conv4:",x.shape)
        x = F.relu(self.conv51(x))
        x = F.relu(self.conv52(x))
        x = F.relu(self.conv52(x))
        #print("After conv5:",x.shape)
        x = F.relu(self.conv61(x))
        x = F.relu(self.conv52(x))
        x = F.relu(self.conv52(x))
        #print("After conv6:",x.shape)
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        #print("After flattening:",x.shape)
        x_mu = self.fc_mu(x)
        #print("x_mu:",x_mu.shape)
        x_logvar = self.fc_logvar(x)
        #print("x_logvar:",x_logvar.shape)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = params['num_channels']
        self.fc = nn.Linear(in_features=params['latent_dims'], out_features=9216)
        self.conv61 = nn.ConvTranspose2d(in_channels=c*2*2*2*2, out_channels=c*2*2*2*2, kernel_size=3, stride=2, padding=0)
        self.conv52 = nn.ConvTranspose2d(in_channels=c*2*2*2*2, out_channels=c*2*2*2*2, kernel_size=2, stride=1, padding=0)
        self.conv51 = nn.ConvTranspose2d(in_channels=c*2*2*2*2, out_channels=c*2*2*2, kernel_size=3, stride=2, padding=0)
        self.conv42 = nn.ConvTranspose2d(in_channels=c*2*2*2, out_channels=c*2*2*2, kernel_size=2, stride=1, padding=0)
        self.conv41 = nn.ConvTranspose2d(in_channels=c*2*2*2, out_channels=c*2*2, kernel_size=3, stride=2, padding=0)
        self.conv32 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=2, stride=1, padding=0)
        self.conv31 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2, kernel_size=3, stride=2, padding=0)
        self.conv22 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c*2, kernel_size=2, stride=1, padding=0)
        self.conv21 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=3, stride=2, padding=0)
        self.conv12 = nn.ConvTranspose2d(in_channels=c, out_channels=c, kernel_size=2, stride=1, padding=0)
        self.conv11 = nn.ConvTranspose2d(in_channels=c, out_channels=3, kernel_size=2, stride=1, padding=0) 

            
    def forward(self, x):
        #print("Decoder:")
        #print("Inputshape:",x.shape)
        x = self.fc(x)
        #print("after FC:",x.shape)
        x = x.view(x.size(0),1024, 3, 3) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print("After unflatten:",x.shape)
        x = F.relu(self.conv52(x))
        x = F.relu(self.conv52(x))
        x = F.relu(self.conv61(x))
        #print("After conv6:",x.shape)
        x = F.relu(self.conv52(x))
        x = F.relu(self.conv52(x))
        x = F.relu(self.conv51(x))
        #print("After conv5:",x.shape)
        x = F.relu(self.conv42(x))
        x = F.relu(self.conv42(x))
        x = F.relu(self.conv41(x))
        #print("After conv4:",x.shape)
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv31(x))
        #print("After conv3:",x.shape)
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv21(x))
        #print("After conv2:",x.shape)
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv11(x))
        #print("After conv1:",x.shape)
        return x
    
class Variational_Autoencoder(nn.Module):
    def __init__(self):
        super(Variational_Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon,latent
    
    def latent_sample(self, mu, logvar):
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
    
    def decode(self,latent):
        x_recon = self.decoder(latent)
        return x_recon



if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Variational_Autoencoder()
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print('Number of parameters: %d' % num_params)
    sample_batch = torch.rand((params['batch_size'],3,params['img_size'],params['img_size']))
    sample_recon_batch,_ = model(sample_batch.cuda())

