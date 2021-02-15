
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from utils import ploty_one_date
from utils import get_labels_one_date
from utils import get_ts_tuple_one_date
from utils import get_ts_str_one_date
#%matplotlib notebook


random.seed(233)

day_init=0   #where do we start?
total_days=len(rank_time_list)#200 #how long do we count?
steps=1      #what strides when we go forward
X=20   #top labels
max_char_bar=50

fig, ax = plt.subplots(figsize=(16, 9))


#l = plt.plot(t, s)

#ax = plt.axis([0,X,-1,1])

x0=np.arange(1, X+1, 1)
xticklabels=["#{}".format(i) for i in range(X+1,0,-1)]
xticklabels[0]=""

y0=ploty_one_date(day_init)[::-1]#reverse!
y0=y0[:X]
ymin=min(y0)
ymax=max(y0)
xlim=(ymin-0.5,10)
ax.set_xlim(xlim)
ax.set_ylim(0,X+1)

# print(x0)
# print(y0)
bar_labels=[] #store data labels on top of each bar
bar_values=[] #store the height of each bar (independent from each tag)
colors = plt.get_cmap('tab20c', X+3)


barcollection = plt.barh(x0, y0)
plt.title("Anime Rank Trends",fontsize=12)
#date_anno = ax.text( (xlim[1]+xlim[0])*0.535,X*1.07, get_ts_str_one_date(day_init), ha='left', va='bottom', fontsize=12)


#hide ticks
ax.yaxis.set_major_locator(plt.MaxNLocator(22))
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=True)
#print(xticklabels)
ax.set_yticklabels(xticklabels,fontsize=14)


ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#ax.spines['top'].set_visible(False)

tagcolor={}#store tag -> color

for i, b in enumerate(barcollection):
    width=b.get_width()
    labels=get_labels_one_date(day_init)[::-1]#reverse!
    tmp = ax.text(width+ymax*0.01, b.get_y()+0.3,  y0[i], fontsize=14)
    bar_labels.append(tmp)
    #tmp = ax.text(width-0.3, b.get_y()+0.1, "   {}".format(labels[i]), fontsize=8)
    tmp = ax.text(xlim[0], b.get_y()+0.1, "   {}".format(labels[i][:max_char_bar]), fontsize=14)
    bar_values.append(tmp)
    #rank count, fixed
    #tmp = ax.text(xlim[0]-0.1, b.get_y()+0.1, "   {}".format(X-i), fontsize=8)
    
    tagname=labels[i]
    if(tagname not in tagcolor):
        #choose a random one
        tagcolor[tagname]=colors(random.randint(0,X-1))
        pass
    b.set_color(tagcolor[tagname])
    #print(tagcolor[tagname])#format: rgba

#data=[k for k in range(X)]
#rounded rect

    

def animate(i_interpol):
    #global tagcolors
    #index need to be int, animate i maybe float
    idx=int(i_interpol)
    y = ploty_one_date(idx)[:X][::-1]#reverse!
    
    
    labels=get_labels_one_date(idx)[:X][::-1]#reverse!
    
    #set height
    ymax=max(y)
    #ymin=min(y)
    xlim=(ymin-0.5,10)
    for i, b in enumerate(barcollection):
        tagname=labels[i]
        if(tagname not in tagcolor):
            #choose a random one
            tagcolor[tagname]=colors(random.randint(0,X-1))
            pass
        
        width=y[i]
        b.set_width(width)
        #b.set_color(colors(X-i))
        b.set_color(tagcolor[tagname])
        
        #score
        txt=bar_labels[i]
        txt.set_x(width+ymax*0.01)
        txt.set_text("{:.2f}".format(y[i]))
        
        #anime name
        txt2=bar_values[i]
        #txt2.set_x(width-0.3)
        txt2.set_x(xlim[0])
        txt2.set_text(labels[i][:max_char_bar])
        colorsum=( sum(tagcolor[tagname][:3]) )/3
        if(colorsum<=0.4):
            txt2.set_color((0.7,0.7,0.7))
            
        else:
            txt2.set_color((0.2,0.2,0.2))
    
        
        ######
    #date_anno.set_x((xlim[1]+xlim[0])*0.515)
    #date_anno.set_text(get_ts_str_one_date(idx))
    plt.title("Anime Rank Trends:{}".format(get_ts_str_one_date(idx)),fontsize=24)
    
    ax.set_xlim(xlim)
    
        
    
    return

# create animation using the animate() function
myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(day_init, total_days, steps), interval=1, repeat=False)
#plt.tight_layout()
plt.show()
# import matplotlib.animation as manimation
# print(manimation.writers.list())



# Writer = manimation.writers['ffmpeg']
# dpi=120
# writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=4000)
# myAnimation.save('im.mp4', writer=writer,dpi=dpi)
# print("done.")