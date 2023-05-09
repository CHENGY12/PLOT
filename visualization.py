import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from torchvision import transforms

# define normalize8 function
def normalize8(I):
    mn = np.min(I)
    mx = np.max(I)
    mx -= mn
    mx = 255 / mx
    ret = np.round((I - mn) * mx).astype(np.uint32)
    return ret

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

#load data
transport_plans = torch.load('path_of_plan') # transport_plans   shape=[N,d,d], e.g. [4,7,7] 
original_image = torch.load('path_of_image') # original_image    shape=[1,3,224,224]
name = 'name_of_class'  # name     class name

#normlize
plans = transport_plans.detach().cpu().numpy().astype(np.float32)
tmp_org =  inv_normalize(original_image[0]).permute(1,2,0) # if the data is from dataloader, please transfer it back
tmp_org = cv2.cvtColor(np.uint8(tmp_org*255),cv2.COLOR_BGR2RGB)

for j in range(plans.shape[0]):
    # obtain path for save
    p = Path(f"results/{name}/{j}")
    p.mkdir(parents=True,exist_ok=True)
    
    # obtain the heatmaps
    tmp = plans[j,:,:]
    tmp = np.uint8(255*(tmp-tmp.min())/(tmp.max()-tmp.min()))
    tmp = cv2.resize(tmp, (224,224))
    
    # visulization 
    viz_atten = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)
    viz_atten_224 = cv2.resize(viz_atten, (224, 224), interpolation=cv2.INTER_CUBIC) 

    # combine with original image
    output = 0.4*viz_atten +tmp_org*0.6
    
    # save for 
    full_path = str((p/f"visualization.jpg").absolute())
    cv2.imwrite(full_path,output)