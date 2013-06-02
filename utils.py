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

