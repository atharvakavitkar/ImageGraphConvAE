# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 08:01:37 2021

@author: Atharva
"""
#import libraries
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import glob
from matplotlib import pyplot as plt
import os
import cv2  
import os



model_name = r'D:/Masters/4/AI Project/model_100.h5'
dataset_dir = r'D:/Masters/4/AI Project/data/hand_drawn_sample/'


class CircuitDataset(Dataset):

    def __init__(self, dataset_dir, transform=None):
        
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self):
        return len(glob.glob(self.dataset_dir+'*'))

    def __getitem__(self, idx):
      ctr = 1
      for file in os.listdir(self.dataset_dir):
        filename = os.fsdecode(file)
        if(ctr==(idx+1)):
          image = Image.open(self.dataset_dir+filename)
          if(self.transform):
                image = self.transform(image)
                return image
        ctr+=1  
      return None
#model class

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c=channel
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=2, stride=1, padding=0) 
        self.conv12 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=2, stride=1, padding=0)

        self.conv21 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=2, padding=0)
        self.conv22 = nn.Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=2, stride=1, padding=0)

        self.conv31 = nn.Conv2d(in_channels=c*2, out_channels=c*2*2, kernel_size=3, stride=2, padding=0)
        self.conv32 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=2, stride=1, padding=0)
        self.fc_mu = nn.Linear(in_features=c*2*2*60*60, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*2*60*60, out_features=latent_dims)
            
    def forward(self, x):
        #print("Encoder:")
        #print(x.shape)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv12(x))
        #print(x.shape)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        #print(x.shape)
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv32(x))
        #print(x.shape)
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        #print(x.shape)
        x_mu = self.fc_mu(x)
        #print(x_mu.shape)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c=channel
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*2*60*60)
        self.conv32 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=2, stride=1, padding=0)
        self.conv31 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2, kernel_size=3, stride=2, padding=0)
        self.conv22 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c*2, kernel_size=2, stride=1, padding=0)
        self.conv21 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=3, stride=2, padding=0)
        self.conv12 = nn.ConvTranspose2d(in_channels=c, out_channels=c, kernel_size=2, stride=1, padding=0)
        self.conv11 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=2, stride=1, padding=0) 

            
    def forward(self, x):
        #print("Decoder:")
        #print("Inputshape:",x.shape)
        x = self.fc(x)
        #print("after FC:",x.shape)
        x = x.view(x.size(0),32, 60, 60) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print("After unflatten:",x.shape)
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv31(x))
        #print("After conv3:",x.shape)
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv21(x))
        #print("After conv2:",x.shape)
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv11(x))
        #print("After conv1:",x.shape)
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        #print("Latent_Dimension:",latent.shape)
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
    
num_epochs = 1
latent_dims = 1000
channel = 8
batchsize = 1
learning_rate = 7e-04
model = VariationalAutoencoder()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

if __name__=='__main__':
    model = torch.load(model_name)
    #model = VariationalAutoencoder()
    transform = transforms.Compose([
                                transforms.Resize((256,256)),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                ])
    circuit_train_dataset = CircuitDataset(dataset_dir+'train/',transform=transform)
    train_loader = torch.utils.data.DataLoader(circuit_train_dataset, batch_size=2, shuffle=True)
    model = model.cuda()
    
    for imagebatch in train_loader:
        imagebatch = imagebatch.cuda()
        recon,latent = model(imagebatch)
        for w in range(0,101):
            morphl = torch.zeros([2, 1000], dtype=torch.float32)
            morphl[0] = (w*0.01*latent[0]) + ((1-(w*0.01))*latent[1])
            morphl[1] = morphl[0]
            morphl = morphl.cuda()
            #print(morphl)
            morph = model.decode(morphl)
            plt.imshow(morph[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.savefig('D:/Masters/4/AI Project/morph/morph'+str(w)+'.png')
        break
    

    frameSize = (500, 500)
    out = cv2.VideoWriter('D:/Masters/4/AI Project/morph/output_video.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, frameSize)

    for filename in glob.glob('D:/Masters/4/AI Project/morph/*.png'):
        if(filename):
            img = cv2.imread(filename)
            out.write(img)

    out.release()
   


    
    
    
    
    
    
    
    
    
    
    
    