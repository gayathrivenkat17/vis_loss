import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import skimage.io
import numpy as np
import torchvision.transforms as transforms
from skimage.transform import resize
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

def train_optic_transform():
    return transforms.Compose([
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def test_optic_transform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def normalize():
    return  transforms.Compose([
    transforms.Resize((450,450)),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def normalize1():
    return  transforms.Compose([
    transforms.ToTensor() # ToTensor : [0, 255] -> [0, 1]
    
])

def normalize2():
    return  transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor() # ToTensor : [0, 255] -> [0, 1]
    
])



class load_train_data_PED(Dataset):

    def __init__(self,k=0):
        super(load_train_data_PED,self).__init__()
        MYDIR='../Data'
        image_data_path = []
        mask_data_path = []
        df=pd.read_csv(r'../CSV_files/train.csv')
        test_df=pd.read_csv(r'../CSV_files/test.csv')
        labels = []
        for i in range(len(df.iloc[:,[0]])):
            image_path = os.path.join(MYDIR,df.iat[i,0])
            image_data_path.append(image_path)
            mask_path = os.path.join(MYDIR,df.iat[i,1])
            mask_data_path.append(mask_path)
            label = df.iat[i,4]
            labels.append(label)
       
        image_data_path=np.array(image_data_path)
        mask_data_path=np.array(mask_data_path)
        labels=np.array(labels)
        data1=np.column_stack((image_data_path,mask_data_path,labels))
        df=pd.DataFrame(data = data1, 
                    columns = ["imageID","maskID","label"])
        df["imageID"]=df["imageID"].astype(str)
        df["maskID"]=df["maskID"].astype(str)
        df["label"]=df["label"].astype(int)
        df_0=df[df["label"]==0]
        df_1=df[df["label"]==1]
        df_0=shuffle(df_0, random_state=0)
        df_0 = df_0.reset_index(drop=True)
        df_0_split = np.array_split(df_0, 10)
        
        merged_df=pd.concat([df_0_split[k], df_1], ignore_index=True)
        self.normalise=normalize()
        self.normalise1=normalize1()
        self.normalise2=normalize2()
        self.train_optic_transform = train_optic_transform() 
        self.tr_df=merged_df
        print("Train data:",self.tr_df["label"].value_counts())

    
    def __len__(self):
        return len(self.tr_df)
    
    def __getitem__(self,id):
        img=Image.open(self.tr_df.iloc[id]["imageID"])
        img = img.convert("RGB")
        img6=self.normalise2(img)
        img=self.normalise(img)
        mask=Image.open(self.tr_df.iloc[id]["maskID"])
        mask=np.array(mask)
        mask[mask!=3]=0
        mask[mask==3]=1
        mask=Image.fromarray(mask)
        mask=self.normalise2(mask)
        mask=mask.numpy()
        mask[mask!=0]=1
        label = torch.tensor(self.tr_df.iloc[id]["label"], dtype=torch.int64)
        return img6,img,mask,label


class load_val_data_PED(Dataset):
    def __init__(self):
        super(load_val_data_PED,self).__init__()
        MYDIR='../Data'
        image_data_path = []
        df=pd.read_csv(r'../CSV_files/train.csv')
        test_df=pd.read_csv(r'../CSV_files/test.csv')
        labels = []
        for i in range(len(df.iloc[:,[0]])):
            image_path = os.path.join(MYDIR,df.iat[i,0])
            image_data_path.append(image_path)
            label = df.iat[i,4]
            labels.append(label)
       
        image_data_path=np.array(image_data_path)
        labels=np.array(labels)
        data1=np.column_stack((image_data_path,labels))
        df=pd.DataFrame(data = data1, 
                    columns = ["imageID","label"])
        df["imageID"]=df["imageID"].astype(str)
        df["label"]=df["label"].astype(int)
        df_0=df[df["label"]==0]
        df_1=df[df["label"]==1]
        df_0=shuffle(df_0, random_state=0)
        df_0 = df_0.reset_index(drop=True)
        df_0_split = np.array_split(df_0, 10)
        
        merged_df=pd.concat([df_0_split[k], df_1], ignore_index=True)
        self.normalise=normalize()
        self.tr_df=merged_df
        print("Train data:",self.tr_df["label"].value_counts())

    
    def __len__(self):
        return len(self.val_df)
    
    def __getitem__(self,id):
        img=Image.open(self.val_df.iloc[id]["imageID"])
        img=self.normalise(img)
        label = torch.tensor(self.val_df.iloc[id]["label"], dtype=torch.int64)
        return img,label
    
    
class load_test_data_PED(Dataset):
    def __init__(self):
        super(load_test_data_PED,self).__init__()
        MYDIR='../Data'
        image_data_path = []
        test_mask_data_path=[]
        df=pd.read_csv(r'../CSV_files/train.csv')
        test_df=pd.read_csv(r'../CSV_files/test.csv')
        test_image_data_path=[]
        test_labels=[]
        for i in range(len(test_df.iloc[:,[0]])):
            image_path = os.path.join(MYDIR,test_df.iat[i,0])
            test_image_data_path.append(image_path)
            mask_path = os.path.join(MYDIR,test_df.iat[i,1])
            test_mask_data_path.append(mask_path)
            label = test_df.iat[i,4]
            test_labels.append(label)
        
        test_image_data_path=np.array(test_image_data_path)
        test_mask_data_path=np.array(test_mask_data_path)
        # print(test_image_data_path)
        test_labels=np.array(test_labels)
        test_data1=np.column_stack((test_image_data_path,test_mask_data_path,test_labels))
        self.test_df=pd.DataFrame(data = test_data1, 
                    columns = ["imageID","maskID","label"])
        self.test_df["imageID"]=self.test_df["imageID"].astype(str)
        self.test_df["maskID"]=self.test_df["maskID"].astype(str)
        self.test_df["label"]=self.test_df["label"].astype(int)
        print("Validation data:",self.test_df["label"].value_counts())
        self.test_optic_transform = test_optic_transform() 
        self.normalise=normalize()
        self.normalise1=normalize1()
        self.normalise2=normalize2()
    
    def __len__(self):
        return len(self.test_df)
    
    def __getitem__(self,id):
        img1=skimage.io.imread(self.test_df.iloc[id]["imageID"])
        m,n=img1.shape
        img2=np.zeros((m,n,3))
        img2[:,:,0]=img1
        img2[:,:,1]=img1
        img2[:,:,2]=img1
        img=Image.fromarray((img2).astype(np.uint8))
        mask=Image.open(self.test_df.iloc[id]["maskID"])
        mask=np.array(mask)
        mask[mask!=3]=0
        mask[mask==3]=1
        mask = Image.fromarray(mask)
        mask=self.normalise2(mask)
        mask=mask.numpy()
        img6=self.normalise2(img)
        img7 = self.normalise(img)
        mask[mask!=0]=1
        label = torch.tensor(self.test_df.iloc[id]["label"], dtype=torch.int64)
        return img6,img7,mask,label
    
    

class load_test_data_PED_seg(Dataset):
    def __init__(self):
        super(load_test_data_PED_seg,self).__init__()
        main_path="../segmentation_Datasets"
        label_path="../segmentation_regions_metadata"
        self.images_path=[]
        self.masks_path=[]
        self.labels=[]
        for i in range(73):
            df=pd.read_csv(os.path.join(label_path,"{}.csv".format(i)))
            img_path=os.path.join(main_path,"Images","{}".format(i))
            mask_path=os.path.join(main_path,"Masks","{}".format(i))
            n=128#len(total_images)
            images=[]
            masks=[]
            label=[]
            for j in range(n):
                masks.append(os.path.join(mask_path,"BScan_{}.jpg".format(j+1)))
                images.append(os.path.join(img_path,"BScan_{}.jpg".format(j+1)))
                label.append(df["Pigmented Epithelial Detachment"][j])
                
            self.masks_path.extend(masks)
            self.images_path.extend(images)
            self.labels.extend(label)
        self.normalise=normalize()
    
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self,id):
        img=skimage.io.imread(self.images_path[id])
        mask=skimage.io.imread(self.masks_path[id])
        m,n=img.shape
        img2=np.zeros((m,n,3))
        img2[:,:,0]=img
        img2[:,:,1]=img
        img2[:,:,2]=img
        img=Image.fromarray((img2).astype(np.uint8))
        img=self.normalise(img)
        label=torch.tensor(self.labels[id],dtype=torch.int64)
            
        return img,label,self.images_path[id],self.masks_path[id]



class load_train_data_SRF(Dataset):

    def __init__(self,k=0):
        super(load_train_data_SRF,self).__init__()
        MYDIR='../Data'
        image_data_path = []
        mask_data_path = []
        df=pd.read_csv(r'../CSV_files/train.csv')
        test_df=pd.read_csv(r'../CSV_files/test.csv')
        labels = []
        for i in range(len(df.iloc[:,[0]])):
            image_path = os.path.join(MYDIR,df.iat[i,0])
            image_data_path.append(image_path)
            mask_path = os.path.join(MYDIR,df.iat[i,1])
            mask_data_path.append(mask_path)
            label = df.iat[i,3]
            labels.append(label)
       
        image_data_path=np.array(image_data_path)
        mask_data_path=np.array(mask_data_path)
        labels=np.array(labels)
        data1=np.column_stack((image_data_path,mask_data_path,labels))
        df=pd.DataFrame(data = data1, 
                    columns = ["imageID","maskID","label"])
        df["imageID"]=df["imageID"].astype(str)
        df["maskID"]=df["maskID"].astype(str)
        df["label"]=df["label"].astype(int)
        df_0=df[df["label"]==0]
        df_1=df[df["label"]==1]
        df_0=shuffle(df_0, random_state=0)
        df_0 = df_0.reset_index(drop=True)
        df_0_split = np.array_split(df_0, 15)
        
        merged_df=pd.concat([df_0_split[k], df_1], ignore_index=True)
        self.normalise=normalize()
        self.normalise1=normalize1()
        self.normalise2=normalize2()
        self.train_optic_transform = train_optic_transform() 
        self.tr_df=merged_df
        print("Train data:",self.tr_df["label"].value_counts())

    
    def __len__(self):
        return len(self.tr_df)
    
    def __getitem__(self,id):
        img=Image.open(self.tr_df.iloc[id]["imageID"])
        img = img.convert("RGB")
        # print("before",img.size)
        img6=self.normalise2(img)
        img=self.normalise(img)
        # print("after",img.size)
        mask=Image.open(self.tr_df.iloc[id]["maskID"])
        mask=np.array(mask)
        mask[mask!=2]=0
        mask[mask==2]=1
        mask=Image.fromarray(mask)
        mask=self.normalise2(mask)
        mask=mask.numpy()
        mask[mask!=0]=1
        # print("second",np.max(mask),np.min(mask))
        label = torch.tensor(self.tr_df.iloc[id]["label"], dtype=torch.int64)
        return img6,img,mask,label



class load_train_data_SRF_old(Dataset):
    def __init__(self,k=0):
        super(load_train_data_SRF,self).__init__()
        MYDIR='../Data'
        image_data_path = []
        df=pd.read_csv(r'../CSV_files/train.csv')
        test_df=pd.read_csv(r'../CSV_files/test.csv')
        labels = []
        for i in range(len(df.iloc[:,[0]])):
            image_path = os.path.join(MYDIR,df.iat[i,0])
            image_data_path.append(image_path)
            label = df.iat[i,3]
            labels.append(label)
    
        image_data_path=np.array(image_data_path)
        labels=np.array(labels)
        data1=np.column_stack((image_data_path,labels))
        df=pd.DataFrame(data = data1, 
                    columns = ["imageID","label"])
        df["imageID"]=df["imageID"].astype(str)
        df["label"]=df["label"].astype(int)
        df_0=df[df["label"]==0]
        df_1=df[df["label"]==1]
        df_0=shuffle(df_0, random_state=0)
        df_0 = df_0.reset_index(drop=True)
        df_0_split = np.array_split(df_0, 15)
        
        merged_df=pd.concat([df_0_split[k], df_1], ignore_index=True)
        self.normalise=normalize()
        self.tr_df=merged_df
        print("Train data:",self.tr_df["label"].value_counts())
    
    def __len__(self):
        return len(self.tr_df)
    
    def __getitem__(self,id):
        img=Image.open(self.tr_df.iloc[id]["imageID"])
        img = img.convert("RGB")
        img=self.normalise(img)
        label = torch.tensor(self.tr_df.iloc[id]["label"], dtype=torch.int64)
        return img,label


class load_val_data_SRF(Dataset):
    def __init__(self):
        super(load_val_data_SRF,self).__init__()
        MYDIR='../Data'
        image_data_path = []
        df=pd.read_csv(r'../CSV_files/train.csv')
        test_df=pd.read_csv(r'../CSV_files/test.csv')
        labels = []
        for i in range(len(df.iloc[:,[0]])):
            image_path = os.path.join(MYDIR,df.iat[i,0])
            image_data_path.append(image_path)
            label = df.iat[i,3]
            labels.append(label)
    
        image_data_path=np.array(image_data_path)
        labels=np.array(labels)
        data1=np.column_stack((image_data_path,labels))
        df=pd.DataFrame(data = data1, 
                    columns = ["imageID","label"])
        df["imageID"]=df["imageID"].astype(str)
        df["label"]=df["label"].astype(int)
        tr_sessions, val_sessions, _, _ = train_test_split(
        df.index.values,
        df.label.values,
        test_size=0.3,
        stratify=df.label.values,
        random_state=6,
    )

        self.tr_df = df.loc[df.index.isin(tr_sessions)]
        self.val_df = df.loc[df.index.isin(val_sessions)]
        print("Validation data:",self.val_df["label"].value_counts())
            
        self.normalise=normalize()
    
    def __len__(self):
        return len(self.val_df)
    
    def __getitem__(self,id):
        img=Image.open(self.val_df.iloc[id]["imageID"])
        img=self.normalise(img)
        label = torch.tensor(self.val_df.iloc[id]["label"], dtype=torch.int64)
        return img,label
    
    


class load_test_data_SRF(Dataset):
    def __init__(self):
        super(load_test_data_SRF,self).__init__()
        MYDIR='../Data'
        image_data_path = []
        test_mask_data_path=[]
        df=pd.read_csv(r'../CSV_files/train.csv')
        test_df=pd.read_csv(r'../CSV_files/test.csv')
        test_image_data_path=[]
        test_labels=[]
        for i in range(len(test_df.iloc[:,[0]])):
            image_path = os.path.join(MYDIR,test_df.iat[i,0])
            test_image_data_path.append(image_path)
            mask_path = os.path.join(MYDIR,test_df.iat[i,1])
            test_mask_data_path.append(mask_path)
            label = test_df.iat[i,3]
            test_labels.append(label)
        
        test_image_data_path=np.array(test_image_data_path)
        test_mask_data_path=np.array(test_mask_data_path)
        # print(test_image_data_path)
        test_labels=np.array(test_labels)
        test_data1=np.column_stack((test_image_data_path,test_mask_data_path,test_labels))
        self.test_df=pd.DataFrame(data = test_data1, 
                    columns = ["imageID","maskID","label"])
        self.test_df["imageID"]=self.test_df["imageID"].astype(str)
        self.test_df["maskID"]=self.test_df["maskID"].astype(str)
        self.test_df["label"]=self.test_df["label"].astype(int)
        print("Validation data:",self.test_df["label"].value_counts())
        self.test_optic_transform = test_optic_transform() 
        self.normalise=normalize()
        self.normalise1=normalize1()
        self.normalise2=normalize2()
    
    def __len__(self):
        return len(self.test_df)
    
    def __getitem__(self,id):
        img1=skimage.io.imread(self.test_df.iloc[id]["imageID"])
        m,n=img1.shape
        img2=np.zeros((m,n,3))
        img2[:,:,0]=img1
        img2[:,:,1]=img1
        img2[:,:,2]=img1
        img=Image.fromarray((img2).astype(np.uint8))
        mask=Image.open(self.test_df.iloc[id]["maskID"])
        mask=np.array(mask)
        mask[mask!=2]=0
        mask[mask==2]=1
        mask = Image.fromarray(mask)
        mask=self.normalise2(mask)
        mask=mask.numpy()
        img6=self.normalise2(img)
        img7 = self.normalise(img)
        mask[mask!=0]=1
        label = torch.tensor(self.test_df.iloc[id]["label"], dtype=torch.int64)
        return img6,img7,mask,label
    
    

class load_test_data_SRF_old(Dataset):
    def __init__(self):
        super(load_test_data_SRF,self).__init__()
        MYDIR='../Data'
        image_data_path = []
        df=pd.read_csv(r'train.csv')
        test_df=pd.read_csv(r'../CSV_files/test.csv')
        test_image_data_path=[]
        test_labels=[]
        for i in range(len(test_df.iloc[:,[1]])):
            image_path = os.path.join(MYDIR,test_df.iat[i,0])
            test_image_data_path.append(image_path)
            label = test_df.iat[i,3]
            test_labels.append(label)
        
        test_image_data_path=np.array(test_image_data_path)
        test_labels=np.array(test_labels)
        test_data1=np.column_stack((test_image_data_path,test_labels))
        self.test_df=pd.DataFrame(data = test_data1, 
                    columns = ["imageID","label"])
        self.test_df["imageID"]=self.test_df["imageID"].astype(str)
        self.test_df["label"]=self.test_df["label"].astype(int)
        print("Test data:",self.test_df["label"].value_counts())
            
        self.normalise=normalize()
    
    def __len__(self):
        return len(self.test_df)
    
    def __getitem__(self,id):
        img1=skimage.io.imread(self.test_df.iloc[id]["imageID"])
        m,n=img1.shape
        img2=np.zeros((m,n,3))
        img2[:,:,0]=img1
        img2[:,:,1]=img1
        img2[:,:,2]=img1
        img=Image.fromarray((img2).astype(np.uint8))
        img=self.normalise(img)
        label = torch.tensor(self.test_df.iloc[id]["label"], dtype=torch.int64)
        return img,label


class load_test_data_SRF_seg(Dataset):
    def __init__(self):
        super(load_test_data_SRF_seg,self).__init__()
        main_path="../segmentation_Datasets"
        label_path="../segmentation_regions_metadata"
        self.images_path=[]
        self.masks_path=[]
        self.labels=[]
        for i in range(73):
            df=pd.read_csv(os.path.join(label_path,"{}.csv".format(i)))
            img_path=os.path.join(main_path,"Images","{}".format(i))
            mask_path=os.path.join(main_path,"Masks","{}".format(i))
            n=128#len(total_images)
            images=[]
            masks=[]
            label=[]
            for j in range(n):
                masks.append(os.path.join(mask_path,"BScan_{}.jpg".format(j+1)))
                images.append(os.path.join(img_path,"BScan_{}.jpg".format(j+1)))
                label.append(df["Subretinal Fluid"][j])
                
            self.masks_path.extend(masks)
            self.images_path.extend(images)
            self.labels.extend(label)
        self.normalise=normalize()
    
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self,id):
        img=skimage.io.imread(self.images_path[id])
        mask=skimage.io.imread(self.masks_path[id])
        m,n=img.shape
        img2=np.zeros((m,n,3))
        img2[:,:,0]=img
        img2[:,:,1]=img
        img2[:,:,2]=img
        img=Image.fromarray((img2).astype(np.uint8))
        img=self.normalise(img)
        label=torch.tensor(self.labels[id],dtype=torch.int64)

            
        return img,label,self.images_path[id],self.masks_path[id]

    



class load_train_data_IRF(Dataset):

    def __init__(self,k=0):
        super(load_train_data_IRF,self).__init__()
        MYDIR='../Data'
        image_data_path = []
        mask_data_path = []
        df=pd.read_csv(r'train.csv')
        test_df=pd.read_csv(r'../CSV_files/test.csv')
        labels = []
        for i in range(len(df.iloc[:,[0]])):
            image_path = os.path.join(MYDIR,df.iat[i,0])
            image_data_path.append(image_path)
            mask_path = os.path.join(MYDIR,df.iat[i,1])
            mask_data_path.append(mask_path)
            label = df.iat[i,2]
            labels.append(label)
       
        image_data_path=np.array(image_data_path)
        mask_data_path=np.array(mask_data_path)
        labels=np.array(labels)
        data1=np.column_stack((image_data_path,mask_data_path,labels))
        df=pd.DataFrame(data = data1, 
                    columns = ["imageID","maskID","label"])
        df["imageID"]=df["imageID"].astype(str)
        df["maskID"]=df["maskID"].astype(str)
        df["label"]=df["label"].astype(int)
        df_0=df[df["label"]==0]
        df_1=df[df["label"]==1]
        df_0=shuffle(df_0, random_state=0)
        df_0 = df_0.reset_index(drop=True)
        df_0_split = np.array_split(df_0, 10)
        
        merged_df=pd.concat([df_0_split[k], df_1], ignore_index=True)
        self.normalise=normalize()
        self.normalise1=normalize1()
        self.normalise2=normalize2()
        self.train_optic_transform = train_optic_transform() 
        self.tr_df=merged_df
        print("Train data:",self.tr_df["label"].value_counts())

    
    def __len__(self):
        return len(self.tr_df)
    
    def __getitem__(self,id):
        img=Image.open(self.tr_df.iloc[id]["imageID"])
        img = img.convert("RGB")
        # print("before",img.size)
        img6=self.normalise2(img)
        img=self.normalise(img)
        # print("after",img.size)
        mask=Image.open(self.tr_df.iloc[id]["maskID"])
        mask=np.array(mask)
        mask[mask!=1]=0
        mask[mask==1]=1
        mask=Image.fromarray(mask)
        mask=self.normalise2(mask)
        mask=mask.numpy()
        mask[mask!=0]=1
        # print("second",np.max(mask),np.min(mask))
        label = torch.tensor(self.tr_df.iloc[id]["label"], dtype=torch.int64)
        return img6,img,mask,label


class load_train_data_IRF_old(Dataset):
    def __init__(self,k=0):
        super(load_train_data_IRF,self).__init__()
        MYDIR='../Data'
        image_data_path = []
        df=pd.read_csv(r'train.csv')
        test_df=pd.read_csv(r'../CSV_files/test.csv')
        labels = []
        for i in range(len(df.iloc[:,[0]])):
            image_path = os.path.join(MYDIR,df.iat[i,0])
            image_data_path.append(image_path)
            label = df.iat[i,2]
            labels.append(label)
    
        image_data_path=np.array(image_data_path)
        labels=np.array(labels)
        data1=np.column_stack((image_data_path,labels))
        df=pd.DataFrame(data = data1, 
                    columns = ["imageID","label"])
        df["imageID"]=df["imageID"].astype(str)
        df["label"]=df["label"].astype(int)
        print("Total data",df["label"].value_counts())
        df_0=df[df["label"]==0]
        df_1=df[df["label"]==1]
        df_0=shuffle(df_0, random_state=0)
        df_0 = df_0.reset_index(drop=True)
        df_0_split = np.array_split(df_0, 5)
        
        merged_df=pd.concat([df_0_split[k], df_1], ignore_index=True)
        self.normalise=normalize()
        self.tr_df=merged_df
        print("Train data:",self.tr_df["label"].value_counts())

    
    def __len__(self):
        return len(self.tr_df)
    
    def __getitem__(self,id):
        img=Image.open(self.tr_df.iloc[id]["imageID"])
        img = img.convert("RGB")
        img=self.normalise(img)
        label = torch.tensor(self.tr_df.iloc[id]["label"], dtype=torch.int64)
        return img,label


class load_val_data_IRF(Dataset):
    def __init__(self):
        super(load_val_data_IRF,self).__init__()
        MYDIR='../Data'
        image_data_path = []
        df=pd.read_csv(r'train.csv')
        test_df=pd.read_csv(r'../CSV_files/test.csv')
        labels = []
        for i in range(len(df.iloc[:,[0]])):
            image_path = os.path.join(MYDIR,df.iat[i,0])
            image_data_path.append(image_path)
            label = df.iat[i,2]
            labels.append(label)
    
        image_data_path=np.array(image_data_path)
        labels=np.array(labels)
        data1=np.column_stack((image_data_path,labels))
        df=pd.DataFrame(data = data1, 
                    columns = ["imageID","label"])
        df["imageID"]=df["imageID"].astype(str)
        df["label"]=df["label"].astype(int)
        tr_sessions, val_sessions, _, _ = train_test_split(
        df.index.values,
        df.label.values,
        test_size=0.3,
        stratify=df.label.values,
        random_state=6,
    )

        self.tr_df = df.loc[df.index.isin(tr_sessions)]
        self.val_df = df.loc[df.index.isin(val_sessions)]
        print("Validation data:",self.val_df["label"].value_counts())
            
        self.normalise=normalize()
    
    def __len__(self):
        return len(self.val_df)
    
    def __getitem__(self,id):
        img=Image.open(self.val_df.iloc[id]["imageID"])
        img = img.convert("RGB")
        img=self.normalise(img)
        label = torch.tensor(self.val_df.iloc[id]["label"], dtype=torch.int64)
        return img,label
    
    

class load_test_data_IRF(Dataset):
    def __init__(self):
        super(load_test_data_IRF,self).__init__()
        MYDIR='../Data'
        image_data_path = []
        test_mask_data_path=[]
        df=pd.read_csv(r'/train.csv')
        test_df=pd.read_csv(r'../CSV_files/test.csv')
        test_image_data_path=[]
        test_labels=[]
        for i in range(len(test_df.iloc[:,[0]])):
            image_path = os.path.join(MYDIR,test_df.iat[i,0])
            test_image_data_path.append(image_path)
            mask_path = os.path.join(MYDIR,test_df.iat[i,1])
            test_mask_data_path.append(mask_path)
            label = test_df.iat[i,2]
            test_labels.append(label)
        
        test_image_data_path=np.array(test_image_data_path)
        test_mask_data_path=np.array(test_mask_data_path)
        # print(test_image_data_path)
        test_labels=np.array(test_labels)
        test_data1=np.column_stack((test_image_data_path,test_mask_data_path,test_labels))
        self.test_df=pd.DataFrame(data = test_data1, 
                    columns = ["imageID","maskID","label"])
        self.test_df["imageID"]=self.test_df["imageID"].astype(str)
        self.test_df["maskID"]=self.test_df["maskID"].astype(str)
        self.test_df["label"]=self.test_df["label"].astype(int)
        print("Validation data:",self.test_df["label"].value_counts())
        self.test_optic_transform = test_optic_transform() 
        self.normalise=normalize()
        self.normalise1=normalize1()
        self.normalise2=normalize2()
    
    def __len__(self):
        return len(self.test_df)
    
    def __getitem__(self,id):
        img1=skimage.io.imread(self.test_df.iloc[id]["imageID"])
        m,n=img1.shape
        img2=np.zeros((m,n,3))
        img2[:,:,0]=img1
        img2[:,:,1]=img1
        img2[:,:,2]=img1
        img=Image.fromarray((img2).astype(np.uint8))
        mask=Image.open(self.test_df.iloc[id]["maskID"])
        mask=np.array(mask)
        mask[mask!=1]=0
        mask[mask==1]=1
        mask = Image.fromarray(mask)
        mask=self.normalise2(mask)
        mask=mask.numpy()
        img6=self.normalise2(img)
        img7 = self.normalise(img)
        mask[mask!=0]=1
        label = torch.tensor(self.test_df.iloc[id]["label"], dtype=torch.int64)
        return img6,img7,mask,label
        
class load_test_data_IRF_old(Dataset):
    def __init__(self):
        super(load_test_data_IRF,self).__init__()
        MYDIR='../Data'
        image_data_path = []
        df=pd.read_csv(r'train.csv')
        test_df=pd.read_csv(r'../CSV_files/test.csv')
        test_image_data_path=[]
        test_labels=[]
        for i in range(len(test_df.iloc[:,[1]])):
            image_path = os.path.join(MYDIR,test_df.iat[i,0])
            test_image_data_path.append(image_path)
            label = test_df.iat[i,2]
            test_labels.append(label)
        
        test_image_data_path=np.array(test_image_data_path)
        test_labels=np.array(test_labels)
        test_data1=np.column_stack((test_image_data_path,test_labels))
        self.test_df=pd.DataFrame(data = test_data1, 
                    columns = ["imageID","label"])
        self.test_df["imageID"]=self.test_df["imageID"].astype(str)
        self.test_df["label"]=self.test_df["label"].astype(int)
        print("Test data:",self.test_df["label"].value_counts())
            
        self.normalise=normalize()
    
    def __len__(self):
        return len(self.test_df)
    
    def __getitem__(self,id):
        img1=skimage.io.imread(self.test_df.iloc[id]["imageID"])
        m,n=img1.shape
        img2=np.zeros((m,n,3))
        img2[:,:,0]=img1
        img2[:,:,1]=img1
        img2[:,:,2]=img1
        img=Image.fromarray((img2).astype(np.uint8))
        img=self.normalise(img)
        label = torch.tensor(self.test_df.iloc[id]["label"], dtype=torch.int64)
        return img,label




class load_test_data_IRF_seg(Dataset):
    def __init__(self):
        super(load_test_data_IRF_seg,self).__init__()
        main_path="../segmentation_Datasets"
        label_path="../segmentation_regions_metadata"
        self.images_path=[]
        self.masks_path=[]
        self.labels=[]
        for i in range(73):
            df=pd.read_csv(os.path.join(label_path,"{}.csv".format(i)))
            img_path=os.path.join(main_path,"Images","{}".format(i))
            mask_path=os.path.join(main_path,"Masks","{}".format(i))
            n=128#len(total_images)
            images=[]
            masks=[]
            label=[]
            for j in range(n):
                masks.append(os.path.join(mask_path,"BScan_{}.tif".format(j+1)))
                images.append(os.path.join(img_path,"BScan_{}.jpg".format(j+1)))
                label.append(df["Intraretinal Fluid"][j])
                
            self.masks_path.extend(masks)
            self.images_path.extend(images)
            self.labels.extend(label)
        self.normalise=normalize()
    
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self,id):
        img=skimage.io.imread(self.images_path[id])
        mask=skimage.io.imread(self.masks_path[id])
        m,n=img.shape
        img2=np.zeros((m,n,3))
        img2[:,:,0]=img
        img2[:,:,1]=img
        img2[:,:,2]=img
        img=Image.fromarray((img2).astype(np.uint8))
        img=self.normalise(img)
        label=torch.tensor(self.labels[id],dtype=torch.int64)
            
        return img,label,self.images_path[id],self.masks_path[id]
    