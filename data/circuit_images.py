from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class CircuitDataset(Dataset):
    def __init__(self, dir, train=True, train_percent=90, size=512):
        assert os.path.exists(dir), "data directory not found"
        self.size = size
        imgs = glob.glob(os.path.join(dir, "*"))
        imgs = [img for img in imgs if "jpeg" in img or "jpg" in img]        
        numbers = np.unique([int(i.split("/")[-1].split("\\")[-1].split("_")[0][1:]) for i in imgs])
        percent = int((train_percent * len(numbers))/100)
        indices = numbers[: percent] if train else numbers[percent:]
        self.data = []
        for ind in indices:
            set = [img for img in imgs if f"C{ind}_" in img]
            self.data.extend(set)
        #self.data = self.data[:16]
        self.transform = self.get_transform(train)

    def get_transform(self, train=True):
        if train:
            return transforms.Compose([
                                #transforms.Grayscale(3),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.5803, 0.5740, 0.5481), (0.0967, 0.1035, 0.1007))
                                # transforms.RandomPerspective(distortion_scale=0.6, p=0.21),
                                # transforms.ColorJitter(brightness=.2, hue=0.5),
                                # transforms.RandomAutocontrast(p=0.2),
                                # transforms.RandomHorizontalFlip(p=0.2),
                                ])

        else:
            return transforms.Compose([
                                #transforms.Grayscale(3),
                                transforms.ToTensor(),
                                ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path)
        image = image.resize((self.size, self.size))
        image = self.transform(image)
        return image


if __name__ == "__main__":
    dataset = CircuitDataset("", train_percent=90, size=256)
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for img in train_dataloader:
        print (img.shape)
        img = img.detach().numpy()[0]
        img = np.transpose(img, (1,2,0))
        print(np.amax(img))
        plt.imshow(img)
        plt.show(block = True)
        break
