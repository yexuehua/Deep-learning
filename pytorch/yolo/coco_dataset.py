from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import os
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CocoDataset(Dataset):
    def __init__(self, path2listFile, transform=None, trans_params=None):
        with open(path2listFile, "r") as file:
            self.path2imgs = file.readlines()

        self.path2labels = [i.replace("image", "label").replace("jpg", "txt")
                            for i in self.path2imgs]

        self.trans_params = trans_params
        self.tranform = transform

    def __len__(self):
        return len(self.path2imgs)

    def __getitem__(self, item):
        path2img = self.path2imgs[item % len(self.path2imgs)].rstrip()
        path2img = "./data/coco" + path2img
        img = Image.open(path2img).convert("RGB")

        path2label = "./data/coco"+self.path2labels[item % len(self.path2labels)].rstrip()

        labels = None

        if os.path.exists(path2label):
            # for instance:
            #   45 0.479492 0.688771 0.955609 0.595500
            #   45 0.736516 0.247188 0.498875 0.476417
            labels = np.loadtxt(path2label).reshape(-1, 5)
        if self.tranform:
            img, labels = self.tranform(img, labels, self.trans_params)
        return img, labels, path2img
