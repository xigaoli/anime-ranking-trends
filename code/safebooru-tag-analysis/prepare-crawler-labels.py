import pandas as pd

booru_csv_path = "./data/safebooru-pic-meta/all_data.csv"

df = pd.read_csv(booru_csv_path)

#df.head()
print(df.shape[0])
# df.sort_values("score",ascending=False, inplace=True)
df = df.sample(frac=1,random_state=233).reset_index(drop=True)

df1=df[:80000]
tagdict={}
def replace_tagname(t):
    if(t=="touhou_(pc-98)"):
        return "touhou"
    
print(df1.shape[0])
for i in range(df1.shape[0]):
    tags = df1.iloc[i]["tags"]
    for t in tags.split(" "):
        if(t not in tagdict):
            tagdict[t]=0
        tagdict[t]+=1
print(len(tagdict))

if("highres" in tagdict):
    del tagdict["highres"]
top_tag_ls =[k for k in tagdict]
top_tag_ls.sort(key=lambda key:tagdict[key],reverse=True)#sory by frequency
for i in range(10):
    print("{}-{}".format(top_tag_ls[i],tagdict[top_tag_ls[i]]))

import json
# labels_path = "./safebooru-labels-topx.json"
# with open(labels_path,"w") as fp1:
#     json.dump(top_tag_ls[:50],fp1)
# print("saved to {}".format(labels_path))

labels_dict_path = "./safebooru-labels-dict.json"

labels_dict={k:i for i,k in enumerate(top_tag_ls[:50])}   #encoding labels to number, for classification

print(labels_dict)
with open(labels_dict_path,"w") as fp1:
    json.dump(labels_dict,fp1)
print("saved to {}".format(labels_dict_path))

