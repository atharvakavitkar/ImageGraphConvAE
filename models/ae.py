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

        self.conv11 = nn.Conv2d(in_channels=3, out_channels=c, kernel_size=3, stride=1, padding=0, bias = False) 
        self.conv12 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=0, bias = False)
        self.conv13 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=0, bias = False)
        self.bn11 = nn.BatchNorm2d(c)
        self.bn12 = nn.BatchNorm2d(c)
        self.bn13 = nn.BatchNorm2d(c)

        self.conv21 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=1, padding=0, bias = False)
        self.conv22 = nn.Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=3, stride=1, padding=0, bias = False)
        self.conv23 = nn.Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=5, stride=2, padding=0, bias = False)
        self.bn21 = nn.BatchNorm2d(c*2)
        self.bn22 = nn.BatchNorm2d(c*2)
        self.bn23 = nn.BatchNorm2d(c*2)

        self.conv31 = nn.Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=3, stride=1, padding=0, bias = False)
        self.conv32 = nn.Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=3, stride=1, padding=0, bias = False)
        self.conv33 = nn.Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=5, stride=2, padding=0, bias = False)
        self.bn31 = nn.BatchNorm2d(c*2)
        self.bn32 = nn.BatchNorm2d(c*2)
        self.bn33 = nn.BatchNorm2d(c*2)

        self.conv41 = nn.Conv2d(in_channels=c*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0, bias = False)
        self.conv42 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0, bias = False)
        self.conv43 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=5, stride=2, padding=0, bias = False)
        self.bn41 = nn.BatchNorm2d(c*2*2)
        self.bn42 = nn.BatchNorm2d(c*2*2)
        self.bn43 = nn.BatchNorm2d(c*2*2)

        # self.conv51 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0, bias = False)
        # self.conv52 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0, bias = False)
        # self.conv53 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0, bias = False)
        # self.bn51 = nn.BatchNorm2d(c*2*2)
        # self.bn52 = nn.BatchNorm2d(c*2*2)
        # self.bn53 = nn.BatchNorm2d(c*2*2)

        # self.conv61 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=7, stride=1, padding=0)
        # self.conv62 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=5, stride=1, padding=0)
        # self.conv63 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=5, stride=1, padding=0)
        # self.bn61 = nn.BatchNorm2d(c*2*2)
        # self.bn62 = nn.BatchNorm2d(c*2*2)
        # self.bn63 = nn.BatchNorm2d(c*2*2)

        # self.conv71 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0)
        # self.conv72 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0)
        # self.conv73 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0)
        # self.conv74 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0)
        # self.conv75 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0)
        # self.conv76 = nn.Conv2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0)
        # self.bn71 = nn.BatchNorm2d(c*2*2)
        # self.bn72 = nn.BatchNorm2d(c*2*2)
        # self.bn73 = nn.BatchNorm2d(c*2*2)
        # self.bn74 = nn.BatchNorm2d(c*2*2)
        # self.bn75 = nn.BatchNorm2d(c*2*2)
        # self.bn76 = nn.BatchNorm2d(c*2*2)


        self.fc = nn.Linear(in_features=160000, out_features=params['latent_dims'])
        #self.bnfc = nn.BatchNorm1d(num_features=params['latent_dims'])



            
    def forward(self, x):
        #print("Encoder:")
        #print("Input:",x.shape)

        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        #print("After conv1:",x.shape)

        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))   
        x = F.relu(self.bn23(self.conv23(x)))  
        #print("After conv2:",x.shape)

        x = F.relu(self.bn31(self.conv31(x)))
        x = F.relu(self.bn32(self.conv32(x)))
        x = F.relu(self.bn33(self.conv33(x)))
        #print("After conv3:",x.shape)

        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn43(self.conv43(x)))
        #print("After conv4:",x.shape)

        # x = F.relu(self.bn51(self.conv51(x)))
        # x = F.relu(self.bn52(self.conv52(x)))
        # x = F.relu(self.bn53(self.conv53(x)))
        # #print("After conv5:",x.shape)

        # x = F.relu(self.bn61(self.conv61(x)))
        # x = F.relu(self.bn62(self.conv62(x)))
        # x = F.relu(self.bn63(self.conv63(x)))
        # #print("After conv6:",x.shape)

        # x = F.relu(self.bn71(self.conv71(x)))
        # x = F.relu(self.bn72(self.conv72(x)))
        # x = F.relu(self.bn73(self.conv73(x)))
        # x = F.relu(self.bn74(self.conv74(x)))
        # x = F.relu(self.bn75(self.conv75(x)))
        # x = F.relu(self.bn76(self.conv76(x)))
        # #print("After conv7:",x.shape)        

        x = torch.flatten(x, 1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        #print("After flattening:",x.shape)

        x = F.relu(self.fc(x))
        #print("After fc:",x.shape)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        c=params['num_channels']

        self.fc = nn.Linear(in_features=params['latent_dims'], out_features=160000)
        #self.bnfc2 = nn.BatchNorm1d(num_features=13300)
        
        self.unflatten = nn.Unflatten(1, torch.Size([c*2*2, 25, 25]))

        # self.conv76 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0)
        # self.conv75 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0)
        # self.conv74 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0)
        # self.conv73 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0)
        # self.conv72 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0)
        # self.conv71 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0)
        # self.bn76 = nn.BatchNorm2d(c*2*2)
        # self.bn75 = nn.BatchNorm2d(c*2*2)
        # self.bn74 = nn.BatchNorm2d(c*2*2)
        # self.bn73 = nn.BatchNorm2d(c*2*2)
        # self.bn72 = nn.BatchNorm2d(c*2*2)
        # self.bn71 = nn.BatchNorm2d(c*2*2)

        # self.conv63 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=5, stride=1, padding=0)
        # self.conv62 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=5, stride=1, padding=0)
        # self.conv61 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=7, stride=1, padding=0)
        # self.bn63 = nn.BatchNorm2d(c*2*2)
        # self.bn62 = nn.BatchNorm2d(c*2*2)
        # self.bn61 = nn.BatchNorm2d(c*2*2)

        # self.conv53 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0, bias = False)
        # self.conv52 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0, bias = False)
        # self.conv51 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0, bias = False)
        # self.bn53 = nn.BatchNorm2d(c*2*2)
        # self.bn52 = nn.BatchNorm2d(c*2*2)
        # self.bn51 = nn.BatchNorm2d(c*2*2)

        self.conv43 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=5, stride=2, padding=0, bias = False)
        self.conv42 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2*2, kernel_size=3, stride=1, padding=0, bias = False)
        self.conv41 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2, kernel_size=3, stride=1, padding=0, bias = False)
        self.bn43 = nn.BatchNorm2d(c*2*2)
        self.bn42 = nn.BatchNorm2d(c*2*2)
        self.bn41 = nn.BatchNorm2d(c*2)

        self.conv33 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c*2, kernel_size=5, stride=2, padding=0, bias = False)
        self.conv32 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c*2, kernel_size=3, stride=1, padding=0, bias = False)
        self.conv31 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c*2, kernel_size=3, stride=1, padding=0, bias = False)
        self.bn33 = nn.BatchNorm2d(c*2)
        self.bn32 = nn.BatchNorm2d(c*2)
        self.bn31 = nn.BatchNorm2d(c*2)

        self.conv23 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c*2, kernel_size=5, stride=2, padding=0, bias = False)
        self.conv22 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c*2, kernel_size=3, stride=1, padding=0, bias = False)
        self.conv21 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=3, stride=1, padding=0, bias = False)
        self.bn23 = nn.BatchNorm2d(c*2)
        self.bn22 = nn.BatchNorm2d(c*2)
        self.bn21 = nn.BatchNorm2d(c)

        self.conv13 = nn.ConvTranspose2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=0, bias = False)
        self.conv12 = nn.ConvTranspose2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=0, bias = False)
        self.conv11 = nn.ConvTranspose2d(in_channels=c, out_channels=3, kernel_size=3, stride=1, padding=0, bias = True)
        self.bn13 = nn.BatchNorm2d(c)
        self.bn12 = nn.BatchNorm2d(c)
        #self.bn11 = nn.BatchNorm2d(3)

    def forward(self, x):
        #print("Decoder:")
        #print("Inputshape:",x.shape)

        x = F.relu(self.fc(x))

        #print("after FC:",x.shape)
        x = self.unflatten(x) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print("After unflatten:",x.shape)
        
        # x = F.relu(self.bn76(self.conv76(x)))
        # x = F.relu(self.bn75(self.conv75(x)))
        # x = F.relu(self.bn74(self.conv74(x)))
        # x = F.relu(self.bn73(self.conv73(x)))
        # x = F.relu(self.bn72(self.conv72(x)))
        # x = F.relu(self.bn71(self.conv71(x)))
        # #print("After conv7:",x.shape)

        # x = F.relu(self.bn63(self.conv63(x)))
        # x = F.relu(self.bn62(self.conv62(x)))
        # x = F.relu(self.bn61(self.conv61(x)))
        # #print("After conv6:",x.shape)
        
        # x = F.relu(self.bn53(self.conv53(x)))
        # x = F.relu(self.bn52(self.conv52(x)))
        # x = F.relu(self.bn51(self.conv51(x)))
        # #print("After conv5:",x.shape)
        
        x = F.relu(self.bn43(self.conv43(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        #print("After conv4:",x.shape)
        
        x = F.relu(self.bn33(self.conv33(x)))
        x = F.relu(self.bn32(self.conv32(x)))
        x = F.relu(self.bn31(self.conv31(x)))
        #print("After conv3:",x.shape)
       
        x = F.relu(self.bn23(self.conv23(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x = F.relu(self.bn21(self.conv21(x)))
        #print("After conv2:",x.shape)

        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = torch.sigmoid(self.conv11(x))
        #print("After conv1:",x.shape)
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
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
    model = Autoencoder()
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print('Number of parameters: %d' % num_params)
    sample_batch = torch.rand((params['batch_size'],3,params['img_size'],params['img_size']))
    sample_recon_batch,_ = model(sample_batch.cuda())

