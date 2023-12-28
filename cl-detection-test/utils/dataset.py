import os
import numpy as np
import pandas as pd
import json
from skimage import io as sk_io
from torch.utils.data import Dataset, DataLoader

class CephXrayDataset(Dataset):
    def __init__(self, txt_file_path, transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_path = txt_file_path
        self.transform = transform
        with open(self.img_path,'r') as f:
            self.data_list = f.readlines()
            self.data_list = [item.strip() for item in self.data_list]  

    def __getitem__(self, index):
        image_path = self.data_list[index] + '.jpg'
        image = sk_io.imread(image_path)

        json_path = self.data_list[index] + '.json'
        with open(json_path,'r') as f:
            info = json.load(f)
        
        landmarks = np.empty((38,2))
        for id,i in enumerate(info['shapes']):
            points = i['points'][0]
            landmarks[id,0] = points[0]
            landmarks[id,1] = points[1]
        
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data_list)

