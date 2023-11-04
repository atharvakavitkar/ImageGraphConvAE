#import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
#from SSIM.ssim import SSIM

with open("./config.yaml", 'r') as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        a=10
        #print(exc)


class Image_Encoder(nn.Module):
    def __init__(self):
        super(Image_Encoder, self).__init__()
        
        c=params['num_channels']

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=c, kernel_size=3, stride=2, padding=0,bias = False) 
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=2, padding=0,bias = False)
        self.bn1 = nn.BatchNorm2d(c)
        self.bn2 = nn.BatchNorm2d(c*2)

        self.fc = nn.Linear(in_features=508032, out_features=params['latent_dims'])
        #self.bnfc = nn.BatchNorm1d(num_features=params['latent_dims'])


    def forward(self, x):
        #print("Image Encoder:")
        #print("Input:",x.shape)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        #print("After conv:",x.shape)


        x = torch.flatten(x, 1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        #print("After flattening:",x.shape)

        x = F.relu(self.fc(x))
        #print("After fc:",x.shape)

        return x

class Image_Decoder(nn.Module):
    def __init__(self):
        super(Image_Decoder, self).__init__()
        
        c=params['num_channels']

        self.fc = nn.Linear(in_features=params['latent_dims'], out_features=508032)
        #self.bnfc = nn.BatchNorm1d(num_features=10160)
        
        self.unflatten = nn.Unflatten(1, torch.Size([c*2, 63, 63]))

        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=3, stride=2, padding=0,bias = False)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=3, kernel_size=3, stride=2, padding=0,bias = True) 
        self.bn2 = nn.BatchNorm2d(c)
        #self.bn1 = nn.BatchNorm2d(3)

    def forward(self, x):
        #print("Image Decoder:")
        #print("Inputshape:",x.shape)

        x = F.relu(self.fc(x))

        #print("after FC:",x.shape)
        x = self.unflatten(x) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print("After unflatten:",x.shape)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv1(x))
        #print("After conv:",x.shape)
        return x



class Graph_Encoder(nn.Module):
    def __init__(self):
        super(Graph_Encoder, self).__init__()

        self.fc = nn.Linear(in_features=8673, out_features=params['latent_dims'])
        #self.bnfc = nn.BatchNorm1d(num_features=params['latent_dims'])

    def forward(self, x):
        #print("Graph Encoder:")
        #print("Input:",x.shape)

        x = torch.flatten(x, 1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        #print("After flattening:",x.shape)

        x = F.relu(self.fc(x))
        #print("After fc:",x.shape)

        return x

class Graph_Decoder(nn.Module):
    def __init__(self):
        super(Graph_Decoder, self).__init__()

        self.fc = nn.Linear(in_features=params['latent_dims'], out_features=8673)
        #self.bnfc = nn.BatchNorm1d(num_features=10160)
        
        self.unflatten = nn.Unflatten(1, torch.Size([1, 177, 49]))

        #self.bn1 = nn.BatchNorm2d(3)

    def forward(self, x):
        #print("Graph Decoder:")
        #print("Inputshape:",x.shape)

        x = F.relu(self.fc(x))

        #print("after FC:",x.shape)
        x = self.unflatten(x) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print("After unflatten:",x.shape)

        return x




    
class Autoencoder_X(nn.Module):
    def __init__(self):
        super(Autoencoder_X, self).__init__()
        self.img_encoder = Image_Encoder()
        self.img_decoder = Image_Decoder()
        self.graph_encoder = Graph_Encoder()
        self.graph_decoder = Graph_Decoder()
        self.latent_space = torch.zeros([params['latent_dims']])
    
    def forward(self, batch, io):
        #print(io)
        if(io == 'img2img'):
            self.latent_space = self.img_encoder(batch)
            x_recon = self.img_decoder(self.latent_space)
        
        elif(io == 'img2graph'):
            self.latent_space = self.img_encoder(batch)
            x_recon = self.graph_decoder(self.latent_space)

        elif(io == 'graph2img'):
            self.latent_space = self.graph_encoder(batch)
            x_recon = self.img_decoder(self.latent_space)
        
        elif(io == 'graph2graph'):
            self.latent_space = self.graph_encoder(batch)
            x_recon = self.graph_decoder(self.latent_space)

        return x_recon,self.latent_space


    def decode(self,latent):
        x_recon = self.decoder(latent)
        return x_recon

if __name__=='__main__':
    import torch.optim as optim
    import random
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Autoencoder_X()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criteria = torch.nn.MSELoss()
    io_types = ['img2img','img2graph','graph2img','graph2graph']
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print('Number of parameters: %d' % num_params)
    graph_batch = torch.rand((params['batch_size'],1,177,49))
    img_batch = torch.rand((params['batch_size'],3,params['img_size'],params['img_size']))

    io = random.choice(io_types)
    #print(io[:3],io[-3:])
    model.train()
    train_batch = img_batch if io[:3] == 'img' else graph_batch
    train_batch = train_batch.to(device)
    target_batch = img_batch if io[-3:] == 'img' else graph_batch
    optimizer.zero_grad()
    train_batch_recon,_ = model(train_batch, io)
    loss = criteria(train_batch_recon, target_batch.to(device))
    loss.backward()
    optimizer.step()

    #print(f'\nTrain Loss: {loss.item()}')

