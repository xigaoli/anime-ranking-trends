#load stuff
import json
import utils

store_path = "./safebooru-date-trend.json"
with open(store_path,"r") as fp1:
    dt_trend = json.load(fp1)
print("loaded {} lines".format(len(dt_trend)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
%matplotlib notebook

day_init=0   #where do we start?
total_days=3771#how long do we count?
steps=10      #what strides when we go forward
X=20   #top labels

fig, ax = plt.subplots(figsize=(10, 5))


#l = plt.plot(t, s)

#ax = plt.axis([0,X,-1,1])

x0=np.arange(0.0, X, 1)

y0=ploty_one_date(day_init)[:X]
ylim=max(y0)
ax.set_ylim(0,ylim*1.2)
# print(x0)
# print(y0)
bar_labels=[] #store data labels on top of each bar
colors = plt.get_cmap('summer', X+3)
date_anno = ax.text(X-1, ylim*1.1, get_ts_str_one_date(day_init), ha='right', va='top', fontsize=16)

print(y0)
barcollection = plt.bar(x0, y0)
plt.title("Safebooru Picture Tag Trends")

for i, b in enumerate(barcollection):
    height=b.get_height()
    labels=get_labels_one_date(day_init)

    tmp = ax.text(b.get_x(), height+max(y0)*0.03, labels[i], fontsize=10)
    bar_labels.append(tmp)

data=[k for k in range(X)]
def animate(i_interpol):
    #index need to be int, animate i maybe float
    idx=int(i_interpol)
    y = ploty_one_date(idx)[:X]
    labels=get_labels_one_date(idx)
    #set height
    ylim=max(y)
    for i, b in enumerate(barcollection):
        
        b.set_height(y[i])
        b.set_color(colors(i))
        
        txt=bar_labels[i]
        txt.set_y(y[i]+ylim*0.03)
        txt.set_text(labels[i])
    date_anno.set_y(ylim*1.1)
    date_anno.set_text(get_ts_str_one_date(idx))
    #adjust range
    ax.set_ylim(0,ylim*1.2)
    
        
    
    return

# create animation using the animate() function
myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(day_init, day_init+total_days, steps), interval=100, repeat=False)

plt.show()

# horizontal bar version

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
%matplotlib notebook


random.seed(233)

day_init=0   #where do we start?
total_days=3771#how long do we count?
steps=5      #what strides when we go forward
X=20   #top labels

fig, ax = plt.subplots(figsize=(5, 3.8))


#l = plt.plot(t, s)

#ax = plt.axis([0,X,-1,1])

x0=np.arange(0.0, X, 1)

y0=ploty_one_date(day_init)[::-1]#reverse!
y0=y0[:X]
ylim=max(y0)
ax.set_xlim(0,ylim*1.2)
# print(x0)
# print(y0)
bar_labels=[] #store data labels on top of each bar
bar_values=[] #store the 
colors = plt.get_cmap('terrain', X+3)

date_anno = ax.text( ylim*0.7,X-0.4, get_ts_str_one_date(day_init), ha='right', va='bottom', fontsize=12)

barcollection = plt.barh(x0, y0)
plt.title("Safebooru Picture Tag Trends",fontsize=10)

#hide ticks
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)

tagcolor={}#store tag -> color

for i, b in enumerate(barcollection):
    width=b.get_width()
    labels=get_labels_one_date(day_init)[::-1]#reverse!
    tmp = ax.text(width+max(y0)*0.03, b.get_y()+0.3,  labels[i], fontsize=9)
    bar_labels.append(tmp)
    tmp = ax.text(0, b.get_y()+0.1, "   {}".format(X-i), fontsize=8)
    
    tagname=labels[i]
    if(tagname not in tagcolor):
        #choose a random one
        tagcolor[tagname]=colors(random.randint(0,X-1))
        pass
    b.set_color(tagcolor[tagname])

#data=[k for k in range(X)]
#rounded rect

    

def animate(i_interpol):
    #global tagcolors
    #index need to be int, animate i maybe float
    idx=int(i_interpol)
    y = ploty_one_date(idx)[:X][::-1]#reverse!
    
    
    labels=get_labels_one_date(idx)[:X][::-1]#reverse!
    
    #set height
    ylim=max(y)
    for i, b in enumerate(barcollection):
        tagname=labels[i]
        if(tagname not in tagcolor):
            #choose a random one
            tagcolor[tagname]=colors(random.randint(0,X-1))
            pass
        
            
        b.set_width(y[i])
        #b.set_color(colors(X-i))
        b.set_color(tagcolor[tagname])
        
        txt=bar_labels[i]
        txt.set_x(y[i]+ylim*0.03)
        txt.set_text(labels[i])
    date_anno.set_x(ylim*0.75)
    date_anno.set_text(get_ts_str_one_date(idx))
    #adjust range
    ax.set_xlim(0,ylim*1.2)
    
        
    
    return

# create animation using the animate() function
myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(day_init, day_init+total_days, steps), interval=100, repeat=False)
plt.tight_layout()
plt.show()

