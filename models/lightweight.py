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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        c=params['num_channels']

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=c, kernel_size=3, stride=2, padding=0,bias = False) 
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=2, padding=0,bias = False)
        #self.conv3 = nn.Conv2d(in_channels=c*2, out_channels=c*2*2, kernel_size=3, stride=2, padding=0,bias = False)
        #self.conv4 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2*2, kernel_size=3, stride=2, padding=0,bias = False)
        self.bn1 = nn.BatchNorm2d(c)
        self.bn2 = nn.BatchNorm2d(c*2)
        #self.bn3 = nn.BatchNorm2d(c*2*2)
        #self.bn4 = nn.BatchNorm2d(c*2*2*2)

        self.fc = nn.Linear(in_features=508032, out_features=params['latent_dims'])
        #self.bnfc = nn.BatchNorm1d(num_features=params['latent_dims'])


    def forward(self, x):
        #print("Encoder:")
        #print("Input:",x.shape)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        #x = F.relu(self.bn4(self.conv4(x)))
        #print("After conv:",x.shape)


        x = torch.flatten(x, 1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        #print("After flattening:",x.shape)

        x = F.relu(self.fc(x))
        #print("After fc:",x.shape)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        c=params['num_channels']

        self.fc = nn.Linear(in_features=params['latent_dims'], out_features=508032)
        #self.bnfc = nn.BatchNorm1d(num_features=10160)
        
        self.unflatten = nn.Unflatten(1, torch.Size([c*2, 63, 63]))

        #self.conv4 = nn.ConvTranspose2d(in_channels=c*2*2*2, out_channels=c*2*2, kernel_size=3, stride=2, padding=0,bias = False)
        #self.conv3 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2, kernel_size=3, stride=2, padding=0,bias = False)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=3, stride=2, padding=0,bias = False)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=3, kernel_size=3, stride=2, padding=0,bias = True) 
        #self.bn4 = nn.BatchNorm2d(c*2*2)       
        #self.bn3 = nn.BatchNorm2d(c*2)
        self.bn2 = nn.BatchNorm2d(c)
        #self.bn1 = nn.BatchNorm2d(3)

    def forward(self, x):
        #print("Decoder:")
        #print("Inputshape:",x.shape)

        x = F.relu(self.fc(x))

        #print("after FC:",x.shape)
        x = self.unflatten(x) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print("After unflatten:",x.shape)

        #x = F.relu(self.bn4(self.conv4(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv1(x))
        #print("After conv:",x.shape)
        return x
    
class LightWeight_Autoencoder(nn.Module):
    def __init__(self):
        super(LightWeight_Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon,latent
    
    
    def decode(self,latent):
        x_recon = self.decoder(latent)
        return x_recon

if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LightWeight_Autoencoder()
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print('Number of parameters: %d' % num_params)
    sample_batch = torch.rand((params['batch_size'],3,params['img_size'],params['img_size']))
    sample_recon_batch,_ = model(sample_batch.cuda())

