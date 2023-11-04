import torch
import torch.optim as optim
from dataset.circuit_combined import CircuitDataset
from torch.utils.data import DataLoader
from SSIM.ssim import SSIM
from trainer import trainer
from models.modelx import Autoencoder_X
from torch.optim.lr_scheduler import StepLR
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import os
import time

def get_hyper():
    with open("./config.yaml", 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

if __name__ == "__main__":
    params = get_hyper()
    for p in params:
        print(p, ': ', params[p])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model = Autoencoder_X()
    print(model)
    model.to(device)

    train_set = CircuitDataset(params['dataset_dir'], train_percent=params['train_percent'], size=params['img_size'])
    valid_set = CircuitDataset(params['dataset_dir'], train=False, train_percent=params['train_percent'], size=params['img_size'])

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True,
                              pin_memory=True, num_workers=params['num_workers'], drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=params['batch_size'], shuffle=True,
                              num_workers=params['num_workers'], drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.95)
    ssim_loss = SSIM()
    io_types = ['img2img','img2graph','graph2img','graph2graph']

    losses = {
        'train': {
            'total': [],
            'i2i': [],
            'i2g': [],
            'g2i': [],
            'g2g': []
        },
        'valid': {
            'total': [],
            'i2i': [],
            'i2g': [],
            'g2i': [],
            'g2g': []
        }
    }

    outputs = {
        'train': [],
        'valid': []
    }

    for epoch in range(params['num_epochs']):
        epoch_start = time.time()

        train_loss, train_i2i_loss, train_i2g_loss, train_g2i_loss, train_g2g_loss, valid_loss, valid_i2i_loss, valid_i2g_loss, valid_g2i_loss, valid_g2g_loss, train_pred, valid_pred = trainer.train(
            model, train_loader, valid_loader, optimizer, ssim_loss, epoch, io_types, device)

        losses['train']['total'].append(train_loss)
        losses['valid']['total'].append(valid_loss)

        losses['train']['i2i'].append(train_i2i_loss)
        losses['train']['i2g'].append(train_i2g_loss)
        losses['train']['g2i'].append(train_g2i_loss)
        losses['train']['g2g'].append(train_g2g_loss)

        losses['valid']['i2i'].append(valid_i2i_loss)
        losses['valid']['i2g'].append(valid_i2g_loss)
        losses['valid']['g2i'].append(valid_g2i_loss)
        losses['valid']['g2g'].append(valid_g2g_loss)

        outputs['train'].append(train_pred)
        outputs['valid'].append(valid_pred)

        scheduler.step()

        if epoch % params['log_interval'] == 0:
            num_samples = len(outputs['train']) if len(outputs['train']) < 6 else 6
            for i in range(1, num_samples):
                train_img = outputs['train'][-i]
                plt.imsave(os.path.join(params['output_dir'], f'{epoch}_train_{i}.jpg'), train_img)

                valid_img = outputs['valid'][-i]
                plt.imsave(os.path.join(params['output_dir'], f'{epoch}_valid_{i}.jpg'), valid_img)

            for mode in ['train', 'valid']:
                plt.title(f'{mode.capitalize()} Learning Curve \n batch_size: {params["batch_size"]}    latent_dims: {params["latent_dims"]}    learning_rate: {params["learning_rate"]}')
                plt.ylabel('Loss')
                plt.xlabel('Epochs')

                for io_type in ['i2i', 'i2g', 'g2i', 'g2g', 'total']:
                    plt.plot(np.arange(len(losses[mode][io_type]), losses[mode][io_type], label=io_type)

                plt.legend()
                plt.savefig(os.path.join(params['output_dir'], f'{mode}_curve.jpg'))
                plt.clf()

        print(f'\nTime taken: {(time.time() - epoch_start):.2f}')
