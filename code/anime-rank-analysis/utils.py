import numpy as np
import datetime

#reuse code from safebooru

#get ranking number (y) for a specific date index
def ploty_one_date(dateidx):
    taglist = rank_time_list[dateidx][1]
    #[[1, 'uniform'], [1, 'toggles'], ...]
    y=np.array([tag[0] for tag in taglist])
    
    return y

#get ranking label (x labels) for a specific date index
def get_labels_one_date(dateidx=0):
    
    taglist = rank_time_list[dateidx][1]
    y=[tag[1] for tag in taglist]
    return y

def get_ts_str_one_date(dateidx=0):
    dt_tuple=rank_time_list[dateidx][0]
    s = datetime.datetime(dt_tuple[0],dt_tuple[1],dt_tuple[2]).strftime('%Y-%m-%d')
    return s
def get_ts_tuple_one_date(dateidx=0):
    dt_tuple=rank_time_list[dateidx][0]
    return dt_tuple


# ax.bar(x,y)
# plt.show()