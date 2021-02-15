import pandas as pd
import utils

#load data

booru_csv_path = "./data/safebooru-pic-meta/all_data.csv"
df = pd.read_csv(booru_csv_path)


print(df.shape[0])

df.sort_values("created_at",ascending=True, inplace=True)
#df = df.sample(frac=1,random_state=233).reset_index(drop=True)
#df.head()



from tqdm import tqdm
import heapq
df1=df

tagdict={}
filtered_words=set(["tagme","highres","long_hair","short_hair"])
#replace synonyms
def replace_tagname(t):
    if(t=="touhou_(pc-98)"):
        return "touhou"
    return t
def should_skip(t):
    if(t in filtered_words):
        return True
    return False

print(df1.shape[0])


#insert new item into toparr, if toparr full, evict min item(top)

X=50   #top X tags
prev_days=14638
prev_year=2010

# weight of each tag connected with time. lets suppose decay is 0.8.
# after x days, the tag weight added from 1 turns to 0.8^x.
# e.g. if we add a tag "touhou" in day 3 (+1). At day 13, it will become (+0.8**10), about 0.107
# after 30 days it will become (+0.8**30), about 0.001, 
# which means it almost contribute nothing compare to the newly arrived 'touhou' tag at day 33(+1).
# this helps old tag to disappear at top X when no more tags added, like LFU
decay_factor=0.97

date_trend=[]
for i in tqdm(range(df1.shape[0])):
    ts = int(df1.iloc[i]["created_at"])
    ts_days = count_days(ts)#days
    
    #decay:
    # all prev item *= decay_num at each new day
    if(prev_days<ts_days):
        dur=ts_days-prev_days
        #print("decay: decay factor for {} days={}".format(dur,(decay_factor**dur)))
        for k in tagdict:
            tagdict[k]*=(decay_factor**dur)
    
    #print(ts_days)
    tags = df1.iloc[i]["tags"]
    for t in tags.split(" "):
        if(should_skip(t)):#skip
            continue
        if(t not in tagdict):
            tagdict[t]=0
        tagdict[t]+=1
    if(ts_days>prev_days):
        
        ls = heapq.nlargest(X,[(tagdict[k],k) for k in tagdict])
        date_trend.append((ts,ls))
    prev_days=ts_days
    #print(ls[:3])
    #ls = sorted([(tagdict[k],k) for k in tagdict],reverse=True)
    #print(ls[:3])
    #print("--")
print(len(tagdict))
print(len(date_trend))
#print(date_trend[:20])


#-------------save to file-------------

import json
print(len(date_trend))
store_path = "./safebooru-date-trend.json"
with open(store_path,"w") as fp1:
    json.dump(date_trend,fp1)
print("OK")