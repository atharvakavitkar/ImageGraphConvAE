
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
import time
from torch import optim
from ssim import SSIM
dataset_dir = r''
torch.cuda.empty_cache()
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
    #print(len(circuit_train_dataset))
    circuit_trainset, circuit_validset = torch.utils.data.random_split(circuit_train_dataset,[900,60])

    train_loader = torch.utils.data.DataLoader(circuit_trainset, batch_size=batchsize, shuffle=True,pin_memory=True,num_workers = 24)
    valid_loader = torch.utils.data.DataLoader(circuit_validset, batch_size=10, shuffle=True)

    test_loader = torch.utils.data.DataLoader(circuit_test_dataset, batch_size=10, shuffle=True)
    
    return train_loader,valid_loader,test_loader



#model class

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c=channel
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

        self.fc_mu = nn.Linear(in_features=c*2*2*2*2*11*11, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*2*2*2*11*11, out_features=latent_dims)
            
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
        c=channel
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*2*2*2*11*11)
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
        x = x.view(x.size(0),128, 11, 11) # unflatten batch of feature vectors to a batch of multi-channel feature maps
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
    
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
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
    

    

#hyperparameters

num_epochs = 20
latent_dims = 1000
channel = 8
learning_rate = 2e-5
batchsize = 10
#model = VariationalAutoencoder()
model = torch.load('')
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
criterion = SSIM()  

model = model.cuda()

#torch.cuda.is_available()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)


def train(train_loader):

    train_loss = []
    training_outputs = []
    valid_loss = []
    valid_outputs = []

    model.train()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss.append(0)
        valid_loss.append(0)
        num_batches = 0
        for image_batch in train_loader:

            optimizer.zero_grad()
            image_batch = image_batch.cuda()
        
            # vae reconstruction
            image_recon_batch,latent_batch = model(image_batch)

            # reconstruction error
            loss = 1 - criterion(image_recon_batch.cuda(), image_batch)

            # backpropagation
            loss.backward()
            optimizer.step()

            train_loss[-1] += loss.item()
            num_batches += 1
      
        train_loss[-1] /= num_batches
        epoch_end = time.time()
        print('Epoch [%d / %d] average reconstruction error: %f \t Time taken:%f' % (epoch+1, num_epochs, train_loss[-1], epoch_end-epoch_start))
        training_outputs.append([image_batch[9], image_recon_batch[9]])

        for image_batch in valid_loader:
            num_batches = 0
            image_batch = image_batch.cuda()
        
            # vae reconstruction
            image_recon_batch,latent_batch = model(image_batch)

            # reconstruction error
            loss = 1 - criterion(image_recon_batch.cuda(), image_batch)

            valid_loss[-1] += loss.item()
            num_batches += 1
        
        valid_loss[-1] /= num_batches
        print(f'\n\nValidation Error:{valid_loss[-1]:.6f} \n\n')
        valid_outputs.append([image_batch[9], image_recon_batch[9]])
        #if(epoch%25 == 0):
    torch.save(model,'' + str(epoch) + ".h5")
    return train_loss,training_outputs,valid_loss,valid_outputs


    



if __name__=='__main__':
    #data_preperation()
    torch.cuda.empty_cache()
    transform = transforms.Compose([
                                transforms.Resize((512,512)),
                                transforms.ToTensor()
                                ])
    
    train_loader,valid_loader,test_loader = get_data_loaders(transform)  #,valid_loader,test_loader

    train_start = time.time()
    train_loss,training_outputs,valid_loss,valid_outputs = train(train_loader)
    train_end = time.time()
    print("Total time taken:",train_end - train_start)

    #sample outputs

    for i in range(len(training_outputs)-5,len(training_outputs)):
        plt.imshow(training_outputs[i][0].permute(1, 2, 0).cpu())
        plt.savefig('')
        plt.imshow(training_outputs[i][1].permute(1, 2, 0).cpu().detach().numpy())
        plt.savefig('')

    #learning curve

    plt.ion()
    fig = plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(train_loss, label = "Training Error")
    valid_loss = [float('nan') if x==0 else x for x in valid_loss]
    plt.plot(valid_loss, label = "Validation Error")
    plt.legend(loc="upper right")
    fig.savefig('')

