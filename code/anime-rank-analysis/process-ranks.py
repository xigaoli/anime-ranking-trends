import json
datapath="./animerank-top.json"

with open(datapath,"r") as fp1:
    rank_time_list_raw=json.load(fp1)
print("loaded {} items".format(len(rank_time_list_raw)))
#print(errlist[-1])

import numpy as np
#select one rec each month
rank_time_list_sel=[]
dates={}
minranklen=1000#minimum number of ranks
#dates[year][month]=?
for i in range(len(rank_time_list_raw)):
    row=rank_time_list_raw[i]
    date=row[0]
    ranks=row[1]
    minranklen=min(len(ranks),minranklen)
    year=date[0]
    month=date[1]
    if(year not in dates):
        dates[year]={}
    if(month not in dates[year]):
        dates[year][month]=1
        rank_time_list_sel.append(row)
#print(minranklen)
# for i in range(3):
#     print(rank_time_list_sel[i][1][0])
rank_time_list=[]
interpol_total_frames=100
interpol_num=30

#interpolation
#row[i] and row[i+1]:
#inbetween_score[i]=np.linspace(sc1,sc2,interpol_num)
#then append 100-interpol_num sc2 to stablize
prev_row=None
for i in range(len(rank_time_list_sel)):
    row=rank_time_list_sel[i]
    if(prev_row is None):
        prev_row=row
        rank_time_list.append(row)
        continue

        
    #create a (minranklen,interpol_total_frames) matrix and transpose
    ls_interpol=[None]*minranklen
    for j in range(minranklen):
        sc1=prev_row[1][j]
        sc2=row[1][j]
        
        
        sclist=list(np.linspace(sc1[0],sc2[0],interpol_num))
        for _ in range(interpol_total_frames-interpol_num):
            sclist.append(sc2[0])
        #if(j==3):
            #print(sc1,sc2)
            #print(sclist)
        ls_interpol[j]=sclist
    #print(ls_interpol)
    for k in range(interpol_total_frames):
        date=row[0]
        ranks=row[1]#ranks = [ [score,title] ]
        new_ranks=[]
        
        for j in range(minranklen):
            title=ranks[j][1]
            sc_new=ls_interpol[j][k]
            #if(j==3):
                #print("sc_new={},title={}".format(sc_new,title))

            new_ranks.append([sc_new,title])
        #print(date)
        rank_time_list.append([date,new_ranks])
        #print(rank_time_list[-1][1][3])
    prev_row=row
            
            
        
# for i in range(0,200,1):
#     print(rank_time_list[i][1][0])
print("selected {} items, interpolated to {} items".format(len(rank_time_list_sel),len(rank_time_list)))


#-------------save to file-------------
import json
datapath="./animerank-interpolated.json"

with open(datapath,"w") as fp1:
    json.dump(rank_time_list,fp1)
print("saved to {}".format(datapath))