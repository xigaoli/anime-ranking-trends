#generate command strings that needs to be executed

#wayback_machine_downloader https://myanimelist.net/topanime.php?type=airing --exact-url --all-timestamps --from=20210104004616

data_path = "./data/myanimelist.net-top/"
#filename="topanime.php_type=airing"
filename="topanime.php"

import os
import bs4
from tqdm import tqdm
import datetime
import re

def get_score_list(source):
    #regex search
    #method 1: scored ([0-9]\.[0-9][0-9]) then $1 for webpage in around 2008-2016
    #method 2: >([0-9]\.[0-9][0-9])< then $1 for webpage in ~2006, and 2017+
    #two methods are mutually exclusive, if one work, the other one won't work, that's great isn't it?
    #method 1
    x = re.findall("[Ss]cored ([0-9]\.[0-9][0-9])", source)
    if(len(x)>10):
        x = [float(item) for item in x]
        return x
    #method 2
    x = re.findall(">\s?([0-9]\.[0-9][0-9])\s?<",source)#\s match possible space
    if(len(x)>10):
        x = [float(item) for item in x]
        return x
    #by default return empty list
    return x

def get_rank_list(fn):
    #format:
    #1. <strong>...</strong> (.text)
    #2. h3, class=hoverinfo_trigger fl-l fs14 fw-b anime_ranking_h3
    #3. a,class=hoverinfo_trigger fs14 fw-b (.text)
    if(os.path.exists(fn)==False):
        fn+="_"
    with open(fn,"rb") as fp1:
        page_source=fp1.read()
        try:
            page_source = page_source.decode('ISO-8859-1')
        except Exception as e:
            #page_source = page_source.decode('ascii')
            print(fn)
            raise e
    soup = bs4.BeautifulSoup(page_source,'html.parser')
    ranklist=[]
    scorelist=get_score_list(page_source)
    #print(soup)
    #method 0
    ls = soup.findAll("a",attrs={"href":re.compile("^anime.php\?id=.*")})
    if(len(ls)>10):
        i=0
        for _,item in enumerate(ls):
            if(item.find("img") is None):
                title = item.text
                sc = scorelist[i]#possible index out of range, let's see]
                
                i+=1
                ranklist.append([sc,title])
        return ranklist
    #method 1
    ls = soup.findAll("strong")
    if(len(ls)>10):
        #print("strong work")
        #score: scored 8.91<br/>
        if(len(ls)>50):
            ls=ls[:50]#prevent capture extra stuff
        for i,item in enumerate(ls):
            title=item.text
            sc = scorelist[i]    
            ranklist.append([sc,title])
        return ranklist
    
    #method 2
    ls = soup.findAll("h3",attrs={"class":"hoverinfo_trigger fl-l fs14 fw-b anime_ranking_h3"})
    if(len(ls)>10):
        if(len(ls)>50):
            ls=ls[:50]#prevent capture extra stuff
        for i,item in enumerate(ls):
            title=item.find("a").text
            sc = scorelist[i]
            ranklist.append([sc,title])
        return ranklist
#     #method 3
#     ls = soup.findAll("a",attrs={"class":"hoverinfo_trigger fs14 fw-b"})
#     #print(ls)
#     if(len(ls)>10):
#         for item in ls:
#             title=item.text
#             #print(title)
#             ranklist.append(title)
#         return ranklist
    #method 4
    ls = soup.findAll("a",attrs={"class":"hoverinfo_trigger"})
    if(len(ls)>10):
        if(len(ls)>50):
            ls=ls[:50]#prevent capture extra stuff
        for i,item in enumerate(ls):
            if(item.find("img") is None):
                title=item.text
                try:
                    sc = scorelist[i]
                except Exception as e:
                    raise e
                ranklist.append([sc,title])
            
    return ranklist

#process data file

errlist=[]
rank_time_list=[]
for foldername_ts in tqdm(os.listdir(data_path)):

    if(foldername_ts.startswith("20") == False):
        continue
    #print(foldername_ts)
    year=int(foldername_ts[:4])
    
    month=int(foldername_ts[4:6])
    day=int(foldername_ts[6:8])
    fn=os.path.join(data_path,foldername_ts,filename).replace("\\","/")
    
    #print(fn)
    s=""
    
    #print(s)
    rank_list = []
    if(os.path.exists(fn)):
        rank_list = get_rank_list(fn)
    
    
    if(len(rank_list)==0):
        errlist.append(fn)
    else:
        ts=(year,month,day)
        rank_time_list.append([ts,rank_list])
        
        #print(fn)
    #break
    
print("Errors:{}".format(len(errlist)))
print("Captured:{}".format(len(rank_time_list)))
#-------------save to file-------------
import json
datapath="./animerank-top.json"

with open(datapath,"w") as fp1:
    json.dump(rank_time_list,fp1)
print("saved to {}".format(datapath))