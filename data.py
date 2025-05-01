import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import os
import glob
from PIL import Image

from utils import *


class Image_dataset(Dataset) :
    def __init__(self, addrs):
        super().__init__()
        self.image_addrs = addrs
        self.transform = image2tensor

        self.x, self.y = [], []

        for dir in os.listdir(addrs) :
            img_path = glob.glob(addrs + "\\" + dir + "\\*.png")

            self.x.extend(img_path)
            self.y.extend([emotion[dir]] * len(img_path))
        
    def __len__(self) :
        return len(self.y)
    
    def __getitem__(self, index):
        image = Image.open(self.x[index])
        image = self.transform(image)

        label = torch.tensor(self.y[index])

        return image, F.one_hot(label, num_classes=8)

if __name__ == "__main__" :
    test_dataset = Image_dataset("C:\\Users\\User\\Desktop\\github\\data\\train")
    print(test_dataset[0])
