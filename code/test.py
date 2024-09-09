import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import InceptionResNetV2
import argparse
import numpy as np
from skimage.filters import threshold_otsu
from torch.autograd import Variable
from dataset import normalize,load_test_data_PED,load_test_data_SRF,load_test_data_IRF,load_test_data_SRF_seg,load_test_data_PED_seg,load_test_data_SRF_seg,load_test_data_PED_seg,load_test_data_IRF_seg
from torchcam.methods import SmoothGradCAMpp,GradCAMpp
from dataset_test import OCTDataset
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix,roc_curve
from torchvision import models
from torchvision import models
import cv2
import torch.nn.functional as F

import timm
import warnings
warnings.filterwarnings('ignore')
parser=argparse.ArgumentParser("Testing")
parser.add_argument("--batchsize",default=1)
parser.add_argument("--weight_path",default="model_PED_BestF1_finetune_dice_loss.pt")
parser.add_argument("--model_name",default="small_Inception ResnetV2_with_pretrained_weights")
parser.add_argument("--disease",default="PED")
from PIL import Image
import time


def IoU(outputs:torch.Tensor, labels:torch.Tensor, threshold, smooth=1e-6):
    assert outputs.size() == labels.size()

    output = torch.where(outputs>=threshold, 1.0, 0.0)
    intersection = torch.sum(output*labels)
    union = torch.sum(output) + torch.sum(labels) - intersection + smooth

    iou = (intersection+smooth)/union

    return iou

def main():
    args=parser.parse_args()
    if torch.cuda.is_available():
        device="cuda:0"
    else:
        device="cpu"

   
    
    
    

    # fixing the model that we want to use
    if args.model_name=="mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=True)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        print('Model Loaded')
    elif args.model_name=="EfficientNet-b5":
        # model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=2)
        model = EfficientNet.from_pretrained('efficientnet-b5').to(device) 
        print('Model Loaded')
    elif args.model_name=="Original_Inception ResnetV2":
        model=timm.create_model('inception_resnet_v2', pretrained=True, num_classes=2)
        print('Model Loaded')
    elif args.model_name=="Small_Inception ResnetV2_with_pretrained_weights":
        model=InceptionResNetV2().to(device) # our proposed model i.e. small Inception ResnetV2 by loading the pretrained weights
        # get the pretrained weights from the original Inception ResnetV2
        model1 =timm.create_model('inception_resnet_v2', pretrained=True, num_classes=2).to(device)
        torch.save(model1.state_dict(),"./model_incresnet.pth")
        pretrained_weights = torch.load("./model_incresnet.pth")
        # Only load the weights for the desired layers
        model_state_dict = model.state_dict()
        pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_state_dict}
        model_state_dict.update(pretrained_weights)
        model.load_state_dict(model_state_dict)
        print("Model Loaded")
    else:
        args.model_name=="Small_Inception ResnetV2_without_pretrained_weights"
        model=InceptionResNetV2().to(device) # our proposed model i.e. small Inception ResnetV2 without loading the pretrained weights
        print('Model Loaded')

    # loading the weights of the model
    model.load_state_dict(torch.load(args.weight_path,map_location=device))
    # loss function 
    weights=[0.5,0.5]
    weights = torch.FloatTensor(weights).to(device)
    loss=torch.nn.CrossEntropyLoss(reduction="mean").to(device)

    # fixing the disease that we want to test
    if args.disease=="PED":
        test_data_load=load_test_data_PED()
    elif args.disease=="SRF":
        test_data_load=load_test_data_SRF()
    elif args.disease=="IRF":
        test_data_load=load_test_data_IRF()
    else:
        print("Invalid Disease Name")
        exit()
    # loading the test data
    val_data_load=OCTDataset(phase='test', pathology='PED')
    test_data=DataLoader(dataset=val_data_load,batch_size=args.batchsize,shuffle=False,pin_memory=True,drop_last=False,num_workers=8)
    
    # testing the model
    for epoch in range(1):
        test_images=tqdm(test_data)
        model.eval()
        output_labels=[]
        true_labels=[]
        images_path=[]
        masks_path=[]
        c=0
        ioutotal=0
        for imgs,masks,labels in test_images:
            masks=masks.to(device)
            test_imgs=Variable(imgs).to(device)
            label=Variable(labels).to(device)
            model_outputs=model(test_imgs).to(device)
            
            _, predicted = torch.max(model_outputs.data, 1)
            output_labels.extend(predicted.data.cpu().numpy())
            true_labels.extend(label.data.cpu().numpy())
            binimgs=[]
            with GradCAMpp(model,target_layer="block8",input_shape=(3,450,450)) as cam_extractor:
            # Preprocess your data and feed it to the model
                out = model(test_imgs[0].unsqueeze(0))
                # Retrieve the CAM by passing the class index and the model output
                activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
                # activation_map = cam_extractor(torch.max(out.data, 1)[1].data.cpu().numpy(), out)
                binimg=activation_map[0].squeeze(0)
                binimg=cv2.resize(binimg.cpu().numpy(),(224,224),interpolation=cv2.INTER_LINEAR)
    
                binimgs.append(torch.from_numpy(binimg).unsqueeze(dim=0))
            # # print(binimgs[0].shape)
            binimgs = torch.cat(binimgs, dim=0).to(device)
            threshold = threshold_otsu(binimgs[0].cpu().numpy())
            if threshold<0.2:
                threshold=1
            binimgs[binimgs>threshold]=1
            binimgs[binimgs<=threshold]=0
            current_iou= IoU(binimgs, masks, threshold).item()
            ioutotal+=current_iou
        # calculating the metrics        
        print("IOU",ioutotal/len(test_data))
        val_f1_score = f1_score(true_labels, output_labels)
        print(confusion_matrix(true_labels,output_labels))
        acc=accuracy_score(true_labels,output_labels) 
        print("precision={0}, recall={1}, f1_score:{2},Accuracy:{3}".format(precision_score(true_labels,output_labels,zero_division='warn'),recall_score(true_labels,output_labels,zero_division='warn'),val_f1_score,acc))


        
if __name__=="__main__":
    main()


# run the code using the following command
# python3 test.py --model_name "Small_Inception ResnetV2_with_pretrained_weights" --disease "PED" --weight_path "../fulltrainedmodels/model_PED_BestF1_finetune_dice_loss.pt"