from random import shuffle
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import InceptionResNetV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import numpy as np
import torchvision.models as models
import torch.optim as opt
from torch.autograd import Variable
from dataset import load_train_data_PED,load_test_data_PED
from torchvision.utils import save_image
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix
import timm
from torchvision import models
import torch.nn.functional as F
from tverskyloss import TverskyLoss, DiceLoss
##############################
from PIL import Image
from torchvision.models import resnet50
import numpy as np
from torchcam.methods import ScoreCAM,LayerCAM,GradCAMpp
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
import torchvision
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
import cv2
from torch.nn.modules.loss import CrossEntropyLoss
##############################


parser=argparse.ArgumentParser("Train PED")
parser.add_argument("--batchsize",default=8)
parser.add_argument("--epochs",default=10)
parser.add_argument("--model_name",default="finetune")

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # else:
            # inputs = torch.sigmoid(inputs)
        # target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
def main():

    args=parser.parse_args()
    if torch.cuda.is_available():
        device="cuda:0"
    else:
        device="cpu"
    best_F1=0
    print("in main")
    

    # fixing the model that we want to use
    if args.model_name == "OpticNet" :
        model = OpticNet(input_size= 224,num_of_classes= 2).to(device) # our proposed model i.e. small Inception ResnetV2 without loading the pretrained weights
        print('Model Loaded')
    elif args.model_name=="mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=True)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        print('Model Loaded')
    elif args.model_name=="EfficientNet-b5":
        model = EfficientNet.from_pretrained('efficientnet-b5').to(device) 
        print('Model Loaded')
    elif args.model_name=="Original_Inception ResnetV2":
        model=timm.create_model('inception_resnet_v2', pretrained=True, num_classes=2)
        print('Model Loaded')
    elif args.model_name=="Small_Inception ResnetV2_with_pretrained_weights":
        print("entering if")
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
    elif args.model_name=="finetune":
        model=InceptionResNetV2().to(device) # our proposed model i.e. small Inception ResnetV2 by loading the pretrained weights
        # get the pretrained weights from the original Inception ResnetV2
        model_path = '../pretrainednormal/model_PED_BestF1_Small_Inception ResnetV2_with_pretrained_weights.pt'  
        checkpoint = torch.load(model_path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint)
        print("Model Loaded")

    elif args.model_name=="Small_Inception ResnetV2_without_pretrained_weights":
        model=InceptionResNetV2().to(device) # our proposed model i.e. small Inception ResnetV2 without loading the pretrained weights
        print('Model Loaded')
    elif args.model_name=="resnet":
        # model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model = models.resnet50(pretrained=True).to(device)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

    elif args.model_name=="vgg16()":

        # Load pre-trained weights
        pretrained_vgg = models.vgg16(pretrained=True)

        # Transfer weights from pretrained_vgg to modified_vgg
        modified_vgg.load_state_dict(pretrained_vgg.state_dict(), strict=False)
        
    else:
        print("Model not found")
        exit()


    model=model.to(device)

    weights=[0.5,0.5] # Here we are using dynamic sampling so weights are equal to 0.5
    weights = torch.FloatTensor(weights).to(device)
    loss=torch.nn.CrossEntropyLoss(reduction="mean").to(device)
    model_opt=opt.Adam(model.parameters(),lr=1e-8, betas=(0.9,0.999))
    scheduler = ReduceLROnPlateau(model_opt, mode='min', factor=0.1, patience=2)

    # validation data is loaded
    val_data_load=load_test_data_PED()
    val_data=DataLoader(dataset=val_data_load,batch_size=1,shuffle=True,pin_memory=True,drop_last=False,num_workers=8)
    loss_train=[]
    loss_valid=[]
    best_epoch=0
    mse_loss = nn.MSELoss()
    dice_loss = DiceLoss(n_classes=1).to(device)
    ce_loss = CrossEntropyLoss()
    ce_loss=torch.nn.CrossEntropyLoss(reduction="mean").to(device)
    tversky = TverskyLoss(alpha=0.3, beta=0.7, smooth=1e-5)
    l2_lambda=0.001
    for epoch in range(int(args.epochs)):
        # print(epoch)
        loss1=[]
        output_labels=[]
        true_labels=[]
        for i  in range(10):
            # here for PED we have done 10 splits so we are loading the data for each split and training the model
            train_data_load=load_train_data_PED(k=i)
            train_data=DataLoader(dataset=train_data_load,batch_size=int(args.batchsize),shuffle=True,pin_memory=True,drop_last=False,num_workers=8)
            real_images=tqdm(train_data)
            model.train()
           
            batch_id=0
            for img,imgs,masks,labels in real_images:
                masks=masks.to(device)
                batch_id+=1
                if batch_id==1:
                    model_opt.zero_grad()
                real_imgs=Variable(imgs).to(device)
                masks=Variable(masks).to(device)
                label=Variable(labels).to(device)
                ldn_input =[]
                model_outputs=model(real_imgs)

                ####uncomment this
                binimgs=[]
                for i in range(real_imgs.shape[0]):
                    with GradCAMpp(model,target_layer="block8",input_shape=(3,450,450)) as cam_extractor:
                    # Preprocess your data and feed it to the model
                        out = model(real_imgs[0].unsqueeze(0))
                        # Retrieve the CAM by passing the class index and the model output
                        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
                        # activation_map = cam_extractor(torch.max(out.data, 1)[1].data.cpu().numpy(), out)
                        binimg=activation_map[0].squeeze(0)
                        binimg=cv2.resize(binimg.cpu().numpy(),(224,224),interpolation=cv2.INTER_LINEAR)
                        binimg=torch.sigmoid(torch.from_numpy(binimg))
                         # binimgs.append(torch.from_numpy(binimg).unsqueeze(dim=0))
                        binimgs.append(binimg.unsqueeze(dim=0))
                        # binimgs.append(torch.from_numpy(binimg).unsqueeze(dim=0))
                
                binimgs = torch.cat(binimgs, dim=0).to(device)
                binimgs=Variable(binimgs,requires_grad=True).to(device)
                l2_reg = torch.tensor(0.).to(device)
        
                for param in model.parameters():
                    l2_reg += torch.norm(param).to(device)
                binimgs=binimgs.unsqueeze(dim=1)
                # model_loss=ce_loss(binimgs, masks)
                # model_loss=mse_loss(binimgs, masks)
                # model_loss=tversky(binimgs, masks)
                # model_loss = dice_loss(binimgs,masks,weight=[1],softmax=False)

                # model_loss =loss(model_outputs,label) + 0.1 * dice_loss(binimgs, masks)
                model_loss =loss(model_outputs,label)
                # print(tversky(binimgs, masks))
                model_loss.backward()
                loss1.append(model_loss.item())
                if batch_id%4==0: # Here batch size is less so by doing this we can show weights are updated after every 4 batches i.e. batch size is 32(4*8)
                    model_opt.step()
                    model_opt.zero_grad()
                

                
                _, predicted = torch.max(model_outputs.data, 1)
                output_labels.extend(predicted.data.cpu().numpy())
                true_labels.extend(label.data.cpu().numpy())
                train_f1_score = f1_score(true_labels, output_labels)
                train_acc=accuracy_score(true_labels,output_labels)
                real_images.set_postfix(desc='[%d/%d] Train_F1: %.4f Accuracy_val: %.4f  loss:  %.4f' % (
                epoch,int(args.epochs),train_f1_score,train_acc,sum(loss1)/len(loss1)))
            
                
        del real_images
        del real_imgs
        loss_train.append(sum(loss1)/len(loss1))

        # validating the model after every epoch and saving the best model based on the F1 score on the validation data 
        val_images=tqdm(val_data)
        model.eval()
        output_labels=[]
        true_labels=[]
        val_loss=[]
        for img,imgs,masks,labels in val_images:
            val_imgs=Variable(imgs).to(device)
            label=Variable(labels).to(device)
        
            model_outputs=model(val_imgs)
            del val_imgs
            
            model_loss=loss(model_outputs,label)
            val_loss.append(model_loss.item())

            _, predicted = torch.max(model_outputs.data, 1)
            output_labels.extend(predicted.data.cpu().numpy())
            true_labels.extend(label.data.cpu().numpy())
        del val_images
        loss_valid.append(sum(val_loss)/len(val_loss))
        val_f1_score = f1_score(true_labels, output_labels)
        print(confusion_matrix(true_labels,output_labels))
        acc=accuracy_score(true_labels,output_labels) 
        print("precision={0}, recall={1}, f1_score:{2},Accuracy:{3}".format(precision_score(true_labels,output_labels,zero_division='warn'),recall_score(true_labels,output_labels,zero_division='warn'),val_f1_score,acc))
        # selecting the best model based on F1 score because F1 score is a better metric for imbalanced datasets
        if(best_F1<val_f1_score):
            best_F1=val_f1_score
            best_epoch=epoch
            torch.save(model.state_dict(),"PED_{}_Diceloss.pt".format(args.model_name))

        print("BEST_F1:{},present_F1,epoch:{},{}".format(best_F1,val_f1_score,best_epoch))

        scheduler.step(sum(val_loss)/len(val_loss))
      

      # saving the loss values for each epoch of training and validation

        np.savetxt("PED_train_loss_{}_Tversky.csv".format(args.model_name),
            loss_train,
            delimiter=", ",
            fmt='% s')
        np.savetxt("PED_val_loss_{}_Tversky.csv".format(args.model_name),
            loss_valid,
            delimiter=", ",
            fmt='% s')


        

        
if __name__=="__main__":
    main()

# run this code for training the model
# python train_PED.py --model_name "Small_Inception ResnetV2_with_pretrained_weights" --epochs 5 --batchsize 8
# python train_PED.py --model_name "finetune" --epochs 5 --batchsize 8
