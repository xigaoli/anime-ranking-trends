
#load
import json
labels_path = "./safebooru-labels-topx.json"
with open(labels_path,"r") as fp1:
    top_tag_ls = json.load(fp1)
print(top_tag_ls)
tagset = set(labels_dict)
def select_criteria(tags):
    taghit=0
    tags = tags.split(" ")
    if(len(tags)<20):#select picture has more than y tags
        return False
    for t in tags:
        if(t in tagset):
            taghit+=1
    if(taghit>20):
        return True
    return False

import pandas as pd

booru_csv_path = "./data/safebooru-pic-meta/all_data.csv"

df = pd.read_csv(booru_csv_path)

#df.head()
print(df.shape[0])
# df.sort_values("score",ascending=False, inplace=True)
df = df.sample(frac=1,random_state=233).reset_index(drop=True)

df2 = df.loc[df['tags'].apply(select_criteria) == True]
print(df2.shape)


import requests
from PIL import Image
from tqdm import tqdm
import os

IMG_SIZE=(512, 384)
def resize_img(src_fn):
    # Opens a image in RGB mode  
    im = Image.open(src_fn) 

    #print("original size:{}".format(im.size))
    newsize = IMG_SIZE
    im1 = im.resize(newsize)
    #print("new size:{}".format(im1.size))
    im1.save(src_fn)

def gen_label_for_img(tags):
    #use labels_dict[tagname]
    tags = tags.split(" ")
    taglabel=[0]*len(labels_dict)
    for t in tags:
        if(t in labels_dict):
            taglabel[labels_dict[t]]=1
    return taglabel
    
headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"}
listfile = "./data/safebooru-pic-meta/list.json"
dataset={}

#for i in tqdm (range (100), desc="Loading..."):
for i in tqdm (range(df2.shape[0]), desc="Loading..."):
    fn = "./data/safebooru-pic-meta/pics/f_{}.jpg".format(i)

    
    url = df2.iloc[i]["sample_url"]
    if(url.startswith("//")):
        url = "http:{}".format(url)
    if(url.startswith("http://")):
        pass
    if(url.startswith("https://")):
        pass
    if(url.endswith(".png")):
        continue
    #print(url)
    tags = df2.iloc[i]["tags"]
    try:
        resp = requests.get(url,headers=headers)
        if(resp.status_code!=200):#error
            continue
        if(len(resp.content)<100):#too little, not image
            continue
        with open(fn,'wb') as fp1:
            fp1.write(resp.content)
        resize_img(fn)
    except Exception as e:
        print("Exception at file {}: {}".format(i,e))
        continue
    
    y=gen_label_for_img(tags)
    dataset[fn]=y
    
print(len(dataset))
print("crawler complete.")