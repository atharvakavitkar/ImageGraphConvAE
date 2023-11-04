from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import xml.etree.ElementTree as ET


class CircuitDataset(Dataset):
    def __init__(self, dir, train=True, train_percent=80, size=224, classes={}):
        assert os.path.exists(dir), "data directory not found"
        self.size = size
        files = glob.glob(os.path.join(dir, "*"))
        img_extns = ['jpg','jpeg','JPG']
        imgs = [file for file in files if any(extn in file for extn in img_extns)]
        self.xmls = [file for file in files if "xml" in file.lower()]
        ckt_ids = np.unique([int(i.split("/")[-1].split("\\")[-1].split("_")[0][1:]) for i in imgs])
        np.random.shuffle(ckt_ids)
        percent = int((train_percent * len(ckt_ids))/100)
        indices = ckt_ids[: percent] if train else ckt_ids[percent:]
        self.data = []
        for ind in indices:
            set = [img for img in imgs if f"C{ind}_" in img]
            self.data.extend(set)
        self.transform = self.get_transform(train)
        cls = self.load_xmls()
        if len(classes) == 0:
            self.classes = cls
        else:
            self.classes = classes


    def load_xmls(self):
        classes = []
        self.meta = {}
        maxx = []
        for xml in self.xmls:
            root = ET.parse(xml).getroot()
            objects = root.findall("object")
            filename = root.findall("filename")[0].text
            size = root.findall("size")[0]
            width = int(size.findall("width")[0].text)
            height = int(size.findall("height")[0].text)
            bboxes = []
            maxx.append(len(objects))
            for object in objects:
                name = object.findall("name")[0].text
                classes.append(name)
                bbox = object.findall("bndbox")[0]
                xmin = int(bbox.findall("xmin")[0].text) / width
                xmax = int(bbox.findall("xmax")[0].text) / width
                ymin = int(bbox.findall("ymin")[0].text) / height
                ymax = int(bbox.findall("ymax")[0].text) / height
                bboxes.append([name, xmin, ymin, xmax, ymax])
            self.meta[filename.split(".")[0]] = [width, height, bboxes]
        classes = np.unique(sorted(classes))
        self.max = max(maxx)
        return classes



    def get_transform(self, train=True):
        if train:
            return transforms.Compose([
                                # transforms.Grayscale(3),
                                transforms.ToTensor(),
                                # transforms.RandomPerspective(distortion_scale=0.6, p=0.21),
                                # transforms.ColorJitter(brightness=.2, hue=0.5),
                                # transforms.RandomAutocontrast(p=0.2),
                                # transforms.RandomHorizontalFlip(p=0.2),
                                ])

        else:
            return transforms.Compose([
                                # transforms.Grayscale(3),
                                transforms.ToTensor(),
                                ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        filename = img_path.split("/")[-1].split("\\")[-1].split(".")[0]
        xml = self.meta[filename]
        bboxes = xml[2]
        width, height = xml[:2]
        meta = np.zeros(shape = (177,self.classes.shape[0] + 4), dtype = np.float64)
        num_bboxes = len(bboxes)
        for meta_idx,bbox  in enumerate(bboxes):
            cls = bbox[0]
            idx = np.where(self.classes == cls)[0][0]
            meta[meta_idx][idx] = 1 #one hot encoding
            meta[meta_idx][-4:] = np.array(bbox[1:]) #[xmin, ymin, xmax, ymax]

        image = Image.open(img_path)
        image = image.resize((self.size, self.size))
        image = np.array(image)
        image = self.transform(image)
        meta = self.transform(meta).float()

        return image, meta, num_bboxes, width, height

if __name__ == "__main__":

    import cv2
    dataset = CircuitDataset("")
    train_dataloader = DataLoader(dataset, batch_size=13, shuffle=False)
    i = 1
    for img, meta, num_bboxes, width, height in train_dataloader:
        print("\n### img ###\n",img.shape)
        print("\n### meta ###\n",meta.shape)
        print("\n### num_bboxes ###\n",num_bboxes.shape)
        print("\n### width ###\n",width.shape)
        print("\n### height ###\n",height.shape)

        img = img.detach().numpy()[0]
        img = np.transpose(img, (1,2,0))
        meta = meta.detach().numpy()[0][0]
        num_bboxes = num_bboxes[0]
        img = (img * 255).astype("uint8")
        img= cv2.UMat(img)
        for i in range(num_bboxes):
            box = meta[i][-4:]
            #print(box)
            xmin = int(box[0] * 224)
            ymin = int(box[1] * 224)
            xmax = int(box[2] * 224)
            ymax = int(box[3] * 224)
            cv2.rectangle(img, (xmin,ymin),(xmax,ymax),(0,255,0),2)
        img = cv2.UMat.get(img)
        plt.imshow(img)
        plt.show()
        break
