# create a dataset class 

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import skimage.io as io
import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

classes = {
    'Intraretinal Fluid': 1,
    'Subretinal Fluid': 2,
    'Pigmented Epithelial Detachment': 3
} 

def transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
def transform_mask():
    return transforms.Compose([
        transforms.ToTensor()
    ])

class OCTDataset(Dataset):
    """OCT dataset."""

    def __init__(self, phase='train',pathology="IRF",preprocess_input=None):
        """
        Args:
            phase (string): train or test or val
            pathology (string): IRF, SRF or PED
            transform (callable, optional): Optional transform to be applied
                on a sample.

        """
        super(OCTDataset, self).__init__()
        self.phase = phase
        self.transform = transform()
        self.transform_mask = transform_mask()
        self.preprocess_input=preprocess_input
        self.main_dir="../CSV_files"
        self.data_dir="../Data"
        # folders_dir=os.listdir(self.main_dir)
        if pathology=="IRF":
                self.pathology="Intraretinal Fluid"
        elif pathology=="SRF":
                self.pathology="Subretinal Fluid"
        elif pathology=="PED":
                self.pathology="Pigmented Epithelial Detachment"

        
        # split the images and masks directories into train and test and val 
        if self.phase=="train":
            self.data=pd.read_csv(os.path.join(self.main_dir,"train.csv"))
            
            print("Number of directories for train: ",len(self.data)/128)
        elif self.phase=="val":
            self.data=pd.read_csv(os.path.join(self.main_dir,"val.csv"))
            print("Number of directories for val: ",len(self.data)/128)
        elif self.phase=="test":
            self.data=pd.read_csv(os.path.join(self.main_dir,"test.csv"))
            print("Number of directories for test: ",len(self.data)/128)
        
        print("For phase: {} value counts of  the pathology {} before Augmenting:{} ".format(self.phase,self.pathology,self.data[self.pathology].value_counts()))
        if self.phase=="train" or self.phase=="val":
            print("Number of images before Augmentation of data: ",len(self.data))
            ratios=self.data[self.pathology].value_counts()
            scale=ratios[0]/ratios[1]
            print("Ratio of 0 to 1 {} for pathology {}: ".format(scale,self.pathology))
            if scale>1:
                    df_1=self.data[self.data[self.pathology]==1]
                    # add df_1 to the original dataframe for scale-1 times
                    l=[df_1]*int(scale-1)
                    self.data=pd.concat([self.data]+l,axis=0)
                    print("Number of images after Augmentation of data: ",len(self.data))   
        else:
            print("Number of images: ",len(self.data))
        # print the value counts of each pathology
        print("For phase: {} valuecounts the pathology {} after Augmenting :{} ".format(self.phase,self.pathology,self.data[self.pathology].value_counts()))
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = io.imread(os.path.join(self.data_dir,self.data.iloc[idx]["ImagePath"]))
        mask = io.imread(os.path.join(self.data_dir,self.data.iloc[idx]["MaskPath"]))

        image=torch.tensor(image)
        image=image.unsqueeze(2)
        image=image.repeat(1,1,3)
        image=image.numpy()

        label = self.data.iloc[idx][self.pathology]
        if self.pathology=="Intraretinal Fluid":
            k=1
        elif self.pathology=="Subretinal Fluid":
            k=2
        elif self.pathology=="Pigmented Epithelial Detachment":
            k=3
        if label==1:
            mask[mask==k]=255
            mask[mask!=255]=0
            mask=cv2.resize(mask,(224, 224))
            mask[mask>0]=255
            mask[mask<255]=0
        else:
            mask=np.zeros((224, 224))
        if self.transform:
            image=Image.fromarray(image)
            mask=Image.fromarray(mask)
            image = self.transform(image)
            mask = self.transform_mask(mask)
            mask = mask.squeeze()
            label=torch.tensor(label)
        if self.preprocess_input:
            image=image.permute(1,2,0)
            image=self.preprocess_input(image)
            image=image.permute(2,0,1)
        return image, mask, label
    


if __name__ == "__main__":
    dataset = OCTDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)


    for image, mask, label in tqdm(loader):

        mask1 = []
        mask = mask[0].permute(1,2,0)

        if len(mask.unique()) > 1:
             mask1 = mask[..., 0]
             break

    

    