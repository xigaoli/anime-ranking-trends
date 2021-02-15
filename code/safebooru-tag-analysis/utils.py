import datetime
import numpy as np
import matplotlib.pyplot as plt

def cvt_time_to_string(epoch):
    
    return datetime.datetime.fromtimestamp(epoch).strftime('%Y-%m-%d')

#how many days since epoch?
def count_days(epoch):
    delta=datetime.datetime.fromtimestamp(epoch) - datetime.datetime(1970,1,1)
    return delta.days

def cvt_time_to_year(epoch):
    #return integer like 2012
    return int(datetime.datetime.fromtimestamp(epoch).strftime('%Y'))




#print(dt_trend[5])

#get ranking number (y) for a specific date index
def ploty_one_date(dateidx):
    taglist = dt_trend[dateidx][1]
    #[[1, 'uniform'], [1, 'toggles'], ...]
    y=np.array([tag[0] for tag in taglist])
    
    return y

#get ranking label (x labels) for a specific date index
def get_labels_one_date(dateidx=0):
    
    taglist = dt_trend[dateidx][1]
    y=[tag[1] for tag in taglist]
    return y
    
def get_ts_str_one_date(dateidx=0):
    epoch=dt_trend[dateidx][0]
    s = cvt_time_to_string(epoch)
    return s
    
#print(get_ts_str_one_date(0))
# ax.bar(x,y)
# plt.show()