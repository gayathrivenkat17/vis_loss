
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from methods import SmoothGradCAMpp,GradCAMpp
from model import InceptionResNetV2
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from methods import SmoothGradCAMpp
import torchvision
import numpy as np
import skimage.io
import timm
import matplotlib.pyplot as plt
from utils import overlay_mask
import cv2

device="cpu"
model=InceptionResNetV2().to(device) # our proposed model i.e. small Inception ResnetV2 by loading the pretrained weights
    # get the pretrained weights from the original Inception ResnetV2
model1 =timm.create_model('inception_resnet_v2', pretrained=True, num_classes=2).to(device)
torch.save(model1.state_dict(),"./model_incresnet_nikhil.pth")
pretrained_weights = torch.load("./model_incresnet_nikhil.pth")
# Only load the weights for the desired layers
model_state_dict = model.state_dict()
pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_state_dict}
model_state_dict.update(pretrained_weights)
model.load_state_dict(model_state_dict)
model.load_state_dict(torch.load("/mnt/sdb1/gayathri/classification_models/model_PED_BestF1_Small_Inception ResnetV2_with_pretrained_weights_nikhil.pt",map_location=device))
# Get your input
img = read_image("/mnt/sdb1/gayathri/classification_models/TP_PED_timm/Images/100.jpg",torchvision.io.ImageReadMode.RGB)
print(img.shape)
# Preprocess it for your chosen model
input_tensor = normalize(resize(img, (450, 450)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
print("input_tensor",input_tensor.shape)
cam_extractor = GradCAMpp(model,target_layer="block8",input_shape=(3,450,450))
with GradCAMpp(model,target_layer="block8",input_shape=(3,450,450)) as cam_extractor:
  # Preprocess your data and feed it to the model
  out = model(input_tensor.unsqueeze(0))
  print("out",out.shape)
  # Retrieve the CAM by passing the class index and the model output
  activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
  print("activation_map",len(activation_map))
  print(activation_map[0].shape)


# Resize the CAM and overlay it
# result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# Display it
# plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
# print(activation_map.size)

img=activation_map[0].squeeze(0)

img=cv2.resize(img.numpy(),(224,224),interpolation=cv2.INTER_LINEAR)
img=np.clip(img * 255, 0, 255).astype(np.uint8)
####check this threshold
img[img>=200]=255  
img[img<200]=0
skimage.io.imsave("gradtest.jpg",img)