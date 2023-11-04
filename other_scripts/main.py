
#import libraries
import torch
import torchvision.transforms as transforms
from dataset.circuit_data import CircuitDataset
import torch.nn as nn
import torch.nn.functional as F
import glob
from matplotlib import pyplot as plt
import os
import time
from SSIM.ssim import SSIM
from models.ae import Autoencoder
from torch.optim.lr_scheduler import StepLR
from models.vae import VariationalAutoencoder
from trainer.trainer import train
import yaml

with open("./hyperparameters.yaml", 'r') as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

for p in params:
    print(p, ' : ', params[p])


def get_data_loaders():

    train_set = CircuitDataset(params['dataset_dir'], train_percent=90, size=params['img_size'])
    test_set = CircuitDataset(params['dataset_dir'], train=False, train_percent=90, size=params['img_size'])

    circuit_trainset, circuit_validset = torch.utils.data.random_split(train_set,[870,97])

    train_loader = torch.utils.data.DataLoader(circuit_trainset, batch_size=params['batch_size'], shuffle=True,pin_memory=True,num_workers = params['num_workers'])
    valid_loader = torch.utils.data.DataLoader(circuit_validset, batch_size=params['batch_size'], shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=params['batch_size'], shuffle=True)
    
    return train_loader,valid_loader,test_loader

def take_snapshot(train_output,valid_output,train_loss,valid_loss):
    #sample outputs
    num_samples = len(train_output) if len(train_output) < 4 else 4
    for i in range(len(train_output)- num_samples,len(train_output)):
        plt.figure()
        plt.imshow(train_output[i][0].permute(1, 2, 0).cpu())
        plt.savefig(os.path.join(params['output_dir'],"train_"+str(len(train_output))+"_in_"+str(i)+".png"))
        plt.imshow(train_output[i][1].permute(1, 2, 0).cpu().detach().numpy())
        plt.savefig(os.path.join(params['output_dir'],"train_"+str(len(train_output))+"_out_"+str(i)+".png"))
    
    if(valid_output):

        for i in range(len(valid_output)-num_samples,len(valid_output)):
            plt.figure()
            plt.imshow(valid_output[i][0].permute(1, 2, 0).cpu())
            plt.savefig(os.path.join(params['output_dir'],"valid_"+str(len(train_output))+"_in_"+str(i)+".png"))
            plt.imshow(valid_output[i][1].permute(1, 2, 0).cpu().detach().numpy())
            plt.savefig(os.path.join(params['output_dir'],"valid_"+str(len(train_output))+"_out_"+str(i)+".png"))

    #learning curve
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(train_loss, label = "Training Error")
    plt.plot(valid_loss, label = "Validation Error")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(params['output_dir'], "learning_curve_"+str(len(train_output))+".png"))


#torch.cuda.is_available()
model = Autoencoder()
#model = torch.load('')
optimizer =  torch.optim.Adam(params=model.parameters(), lr=params['learning_rate'])
scheduler = StepLR(optimizer, step_size=5, gamma=0.95)
criterion = SSIM()
model.cuda()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)


def train(train_loader,valid_loader):

    train_loss = []
    train_output = []
    valid_loss = []
    valid_output = []

    model.train()
    for epoch in range(params['num_epochs']):
        epoch_start = time.time()

        num_batches = 0
        valid_loss.append(0)
        for image_batch in valid_loader:
            if(len(image_batch)>1):

                image_batch = image_batch.cuda()
            
                # vae reconstruction
                model, train_loss, test_loss, preds = trainer.train(model, train_loader,
                                                         test_loader, optimizer,
                                                         loss, epoch, device, maximize)


                # reconstruction error
                loss = 1 - criterion(image_recon_batch.cuda(), image_batch)

                valid_loss[-1] += loss.item()
                num_batches += 1
        valid_loss[-1] /= num_batches

        train_loss.append(0)
        num_batches = 0
        for image_batch in train_loader:

            optimizer.zero_grad()
            image_batch = image_batch.cuda()

            # vae reconstruction
            image_recon_batch = model(image_batch)

            # reconstruction error
            loss = 1 - criterion(image_recon_batch.cuda(), image_batch)

            # backpropagation
            loss.backward()
            optimizer.step()

            train_loss[-1] += loss.item()
            num_batches += 1

        train_loss[-1] /= num_batches
        scheduler.step()
        
        print('\n\nEpoch [%d / %d] average reconstruction error: %f\n' % (epoch+1, params['num_epochs'], train_loss[-1]))
        train_output.append([image_batch[-1], image_recon_batch[-1]])
        valid_output.append([image_batch[-1], image_recon_batch[-1]])
        print(f'Validation Error:{valid_loss[-1]:.6f} \t\t\t\t Time taken:{(time.time() - epoch_start):.2f}') #Validation Error:{valid_loss[-1]:.6f} \t\t\t\t 

        if(epoch and epoch%11 == 0):
            print('Saving checkpoint data...')
            torch.save(model,os.path.join(params['output_dir'],str(epoch)+".h5"))
            take_snapshot(train_output,valid_output,train_loss,valid_loss)

    return train_loss,train_output,valid_loss,valid_output


    



if __name__=='__main__':
    #data_preperation()
    train_start = time.time()
    torch.cuda.empty_cache()

    train_loader,valid_loader,test_loader = get_data_loaders()  #,valid_loader,test_loader
    train_loss,train_output,valid_loss,valid_output = train(train_loader,valid_loader)
    print("Total time taken:",time.time() - train_start)

