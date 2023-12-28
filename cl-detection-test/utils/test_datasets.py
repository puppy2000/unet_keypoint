from dataset import CephXrayDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import skimage as ski
from skimage import io
from skimage import draw
from torchvision import transforms
from tranforms import Rescale, RandomHorizontalFlip, ToTensor

train_path = '/data/detection/cl-det-fix/train.txt'
val_path = '/data/detection/cl-det-fix/val.txt'
test_path = '/data/detection/cl-det-fix/test.txt'

train_transform = transforms.Compose([Rescale(output_size=(512, 512)),
                                    # RandomHorizontalFlip(p=0.5),
                                      ToTensor()])

datasets = CephXrayDataset(train_path,transform=train_transform)

# datasets = CephXrayDataset(train_path)

loader = DataLoader(datasets,batch_size=1)

for d in tqdm.tqdm(loader):
    # image, heatmap = d['image'], d['landmarks']

    # image = image[0,...].numpy()
    # heatmap = heatmap[0,...].numpy()
    # radius = 7
    # for i in range(38):
    #     y,x = heatmap[i,0],heatmap[i,1]
    #     rr, cc = draw.disk(center=(int(x), int(y)), radius=radius, shape=image.shape)
    #     image[rr,cc,:] = [255,0,0]
    
    # print(image[int(x),int(y),:])
    # io.imsave('/home/zt/point.png',image)

    # for j in range(38):
    #     image, heatmap = d['image'], d['heatmap']
    #     heatmap = heatmap[0,j,...].numpy()
    #     heatmap = heatmap * 255
    #     heatmap = heatmap[:,:,np.newaxis]
    #     heatmap = np.repeat(heatmap,repeats=3,axis=-1)
    #     heatmap = np.uint8(heatmap)

    #     io.imsave('/home/zt/3.png',heatmap)

    # image = image[0,...].numpy()
    # image = image * 255
    # image = image.transpose((1,2,0))    
    # image = np.uint8(image)
    
    # io.imsave('/home/zt/2.png',image)

    image, heatmap = d['image'], d['heatmap']

    heatmap = heatmap[0,...].numpy()

    coord_list = []

    for i in range(38):
      heatmap_i = heatmap[i,...]
      maximum = heatmap_i.max()
      coord = np.where(heatmap_i == maximum)
      coord_list.append(coord)
      print(heatmap_i[coord[0],coord[1]])

    radius = 3
    
    image = image[0,...]
    image = image.permute(1,2,0)
    image = image * 255
    image = image.numpy()

    for i in range(38):
        x,y = coord_list[i][0],coord_list[i][1]
        rr, cc = draw.disk(center=(int(x), int(y)), radius=radius, shape=image.shape)
        image[rr,cc,:] = [255,0,0]

    image = np.uint8(image)
    io.imsave('/home/zt/2.png',image)
     
