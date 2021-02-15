import torch
from PIL import Image
import torchvision
import numpy as np
import utils
import json
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE=(300,300)
manifest_path = "./data/safebooru-pic-meta/list.json"
label_path = "./safebooru-labels-dict.json"
model_path = "./safebooru-anime_vgg16.pth"
vgg16 = torch.load(model_path, map_location=torch.device(device))
with open(label_path,"r") as fp1:
    classes = json.load(fp1)
    classes = {k:int(v) for k,v in classes.items()}
    classes_rev = {int(v):k for k,v in classes.items()}
print(classes_rev)
def recognize_img(img_paths):
    imgs=[]
    for img_path in img_paths:
        img_bytes = Image.open(img_path)
        img = torchvision.transforms.functional.resize(img=img_bytes,size=IMG_SIZE)#resize img
        img=(np.array(img)/255.0).astype(np.float32)
        imgs.append(img)
    imgs = np.asarray(imgs)
    #subset=np.append(subset,item[0])
    imgs_tensor = torch.tensor(imgs)
    imgs_tensor = imgs_tensor.to(device)
    b = imgs_tensor.permute(0,3,1,2)

    print(b.shape)
    outputs = vgg16(b)
    outputs = torch.max(outputs, 1)[1].data.cpu().numpy() #convert into array
    print(outputs)
    
    utils.show_imgs(imgs,real_labels=None,pred_labels=outputs,classes_rev=classes_rev)
    
img_paths = ["./data/etc_imgs/dd{}.jpg".format(i) for i in range(1,5)]
recognize_img(img_paths)