from __future__ import division
import time

def show_status(info, cur=None, total=None):
    """
    args:
        info: show info
        cur: if cur is None, then only show info

    usage:
        show_status('status: ', 1, 10)
        show_status('finish parser!')
    """
    _time = time.strftime("%d-%H:%M:%S")
    if cur is None:
        print info, _time
        return
    step = int(total / 10)
    if cur % step == 0:
        status = cur/total
        print info, status, 
        print _time

def cal_map(key_list, res_list):
    key_set = set(key_list)
    res = 0
    hit_count = 0
    for i,res in enumerate(res_list):
        if res in key_set:
            hit_count += 1
            res += hit_count / (i+1)
            if hit_count == len(key_set):
                break
    return res / len(key_set)
