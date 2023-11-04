import torch
import random
import cv2
import numpy as np

def train(model, train_loader, valid_loader, optimizer, ssim_loss, epoch, io_types, device='cpu'):
    """
    Train the autoencoder model on the given data loaders.

    Args:
        model (nn.Module): The autoencoder model to train.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): The optimizer used for training.
        ssim_loss (SSIM): The SSIM loss function.
        epoch (int): The current training epoch.
        io_types (list): List of input-output types.
        device (str): The device to use for training ('cpu' or 'cuda').

    Returns:
        Tuple: A tuple containing train and validation losses and visualizations.
    """
    def get_loss_img(input_img, target_img):
        # Compute loss based on SSIM between input and target images
        loss = 1 - ssim_loss(input_img, target_img)
        return loss

    def calculate_loss(train_batch, train_batch_recon, io_train):
        # Calculate the loss based on the input-output type
        if io_train[-3:] == 'img':
            return get_loss_img(train_batch_recon, train_batch)
        else:
            return get_loss_img(train_batch_recon, train_batch)

    losses = {
        'train': {
            'total': 0,
            'i2i': 0,
            'i2g': 0,
            'g2i': 0,
            'g2g': 0
        },
        'valid': {
            'total': 0,
            'i2i': 0,
            'i2g': 0,
            'g2i': 0,
            'g2g': 0
        }
    }

    i2i_ctr = i2g_ctr = g2i_ctr = g2g_ctr = 0

    for img_batch, graph_batch, num_bboxes, width, height in train_loader:
        random.shuffle(io_types)
        for io_train in io_types:
            model.train()
            train_batch = img_batch if io_train[:3] == 'img' else graph_batch
            train_batch = train_batch.to(device)
            optimizer.zero_grad()
            train_batch_recon, _ = model(train_batch, io_train)
            loss = calculate_loss(train_batch, train_batch_recon, io_train)

            losses['train'][io_train] += loss.item()
            losses['train']['total'] += loss.item()

            if io_train == 'img2img':
                i2i_ctr += 1
            elif io_train == 'img2graph':
                i2g_ctr += 1
            elif io_train == 'graph2img':
                g2i_ctr += 1
            elif io_train == 'graph2graph':
                g2g_ctr += 1

            loss.backward()
            optimizer.step()

    # Calculate average losses
    for mode in ['train', 'valid']:
        for io_type in io_types:
            losses[mode][io_type] /= len(train_loader)
        losses[mode]['total'] /= (4 * len(train_loader))

    # Remaining code for validation and visualization

    return (
        losses['train']['total'],
        losses['train']['i2i'],
        losses['train']['i2g'],
        losses['train']['g2i'],
        losses['train']['g2g'],
        losses['valid']['total'],
        losses['valid']['i2i'],
        losses['valid']['i2g'],
        losses['valid']['g2i'],
        losses['valid']['g2g'],
        train_pred,
        valid_pred
    )

def get_bboximg(img, graph, num_bboxes):
    """
    Generate an image with bounding boxes based on graph information.

    Args:
        img (Tensor): Input image.
        graph (Tensor): Graph information.
        num_bboxes (int): Number of bounding boxes.

    Returns:
        ndarray: Image with bounding boxes.
    """
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    graph = graph[-1].cpu().detach().numpy()
    img = (img * 255).astype("uint8")
    img = cv2.UMat(img)

    for i in range(num_bboxes):
        box = graph[i][-4:]
        xmin = int(box[0] * 255)
        ymin = int(box[1] * 255)
        xmax = int(box[2] * 255)
        ymax = int(box[3] * 255)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    img = cv2.UMat.get(img)
    return img
