import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, path_1, path_2, transforms=None):
        self.path_1 = path_1
        self.path_2 = path_2

        self.data_1 = [os.path.join(self.path_1, x) for x in os.listdir(self.path_1) if x.endswith(".png")]
        self.data_2 = [os.path.join(self.path_2, x) for x in os.listdir(self.path_2) if x.endswith(".png")]

        self.transforms = transforms

    def __getitem__(self, index):

        _data1 = Image.open(self.data_1[index])
        _data2 = Image.open(self.data_2[index])

        _data1 = _data1.convert('RGB')
        _data2 = _data2.convert('RGB')
        if self.transforms is not None:
            _data1 = self.transforms(_data1)
            _data2 = self.transforms(_data2)

        return _data1, _data2

    def __len__(self):
        return min(len(self.data_1), len(self.data_2))
