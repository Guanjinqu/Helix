from torch.utils.data import Dataset

from PIL import Image 
import os
from torchvision import transforms

import numpy as np

class MyData(Dataset):

    def __init__(self,root_dir):
        
        self.root_dir = root_dir

        self.img_path = os.listdir(self.root_dir)
        self.transform_1v = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform_3v = transforms.Compose([
                                transforms.ToTensor(),
                                #transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def out(self):
        print("no label")
    def __getitem__(self, index) :
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir,img_name)
        img = Image.open(img_item_path)
        img_np = np.array(img)
        #print(img_np.shape)
        if img_np.shape == (48,48):

            img = self.transform_1v(img)
        else:
            img = self.transform_3v(img)

        #print(np.array(img).shape)
        label = "no label"
        return img,label
    
    def __len__(self):
        return len(self.img_path) 
    
    

if __name__ == "__main__" :
    data_dir = "train/" 

    dataset = MyData(data_dir)

    for i in range(10000):
        img,target = dataset[i]

        #print(img,target,i)