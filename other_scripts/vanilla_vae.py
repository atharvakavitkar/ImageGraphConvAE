
#import libraries
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import glob
#from matplotlib import pyplot as plt
import os
import re
import shutil

dataset_dir = r''
# #data preperation
# print('It works!!')
# def data_preperation():
    
#     image_formats = ['.jpg','.jpeg','.JPG','.png','.JPEG','.PNG']
#     ctr=0
#     for file in os.listdir(dataset_dir):
#         filename = os.fsdecode(file)
#     for ext in image_formats:
#         if filename.endswith(ext):
#             ctr+=1
#             circuit = int(re.search(r'\d+', filename).group())
#             if(circuit<121):
#                 shutil.move(dataset_dir+filename, dataset_dir+filename)
#             else:
#                 shutil.move(dataset_dir+filename, dataset_dir+filename)
#     print(ctr,"files moved.")

#dataset class

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


def get_data_loaders(transform):
    
    circuit_train_dataset = CircuitDataset(dataset_dir+'train/',transform=transform)
    circuit_test_dataset = CircuitDataset(dataset_dir+'test/',transform=transform)
    print(len(circuit_train_dataset))
    circuit_trainset, circuit_validset = torch.utils.data.random_split(circuit_train_dataset,[900,60])

    train_loader = torch.utils.data.DataLoader(circuit_trainset, batch_size=10, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(circuit_validset, batch_size=10, shuffle=True)

    test_loader = torch.utils.data.DataLoader(circuit_test_dataset, batch_size=10, shuffle=True)
    
    return train_loader,valid_loader,test_loader



#model class

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c=channel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=4, padding=0) 
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=4, padding=0)
        self.fc_mu = nn.Linear(in_features=c*2*32*32, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*32*32, out_features=latent_dims)
            
    def forward(self, x):
        # print("Encoder:")
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
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
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*32*32)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=4, padding=0)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=4, padding=0)
            
    def forward(self, x):
        #print("Decoder:")
        #print("Inputshape:",x.shape)
        x = self.fc(x)
        #print("after FC:",x.shape)
        x = x.view(x.size(0),16, 32, 32) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print("After unflatten:",x.shape)
        x = F.relu(self.conv2(x))
        #print("After conv2:",x.shape)
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
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
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    

#hyperparameters

num_epochs = 20
latent_dims = 1000
channel = 8
variational_beta = 1
learning_rate = 3e-03
model = VariationalAutoencoder()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

torch.cuda.is_available()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)


def train(train_loader):

    training_outputs = []
    train_loss_avg = []
    model.train()
    for epoch in range(num_epochs):
        train_loss_avg.append(0)
        num_batches = 0
        for i,image_batch in enumerate(train_loader, 0):
    
          image_batch = image_batch.to(device)
    
          # vae reconstruction
          image_batch_recon, latent_mu, latent_logvar = model(image_batch)
    
          # reconstruction error
          loss = criterion(image_batch_recon, image_batch)
    
          # backpropagation
          for param in model.parameters():
            param.grad = None
          loss.backward()
          optimizer.step()
    
          train_loss_avg[-1] += loss.item()
          num_batches += 1
      
        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))
        training_outputs.append([image_batch[9], image_batch_recon[9]])
    



if __name__=='__main__':
    #data_preperation()
    transform = transforms.Compose([
                                transforms.Resize((512,512)),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    train_loader,valid_loader,test_loader = get_data_loaders(transform)  
    
    train(train_loader)

# #learning curve

# plt.ion()
# fig = plt.figure()
# plt.plot(train_loss_avg)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()

# #sample outputs

# for i in range(15,len(training_outputs)):
#   plt.imshow(training_outputs[i][0].permute(1, 2, 0))
#   plt.show()
#   plt.imshow(training_outputs[i][1].permute(1, 2, 0).detach().numpy())
#   plt.show()