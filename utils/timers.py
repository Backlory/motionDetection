from functools import wraps
import time
from utils.mics import colorstr

def fun_run_time(func):
    '''
    装饰器，用于获取函数的执行时间
    放在函数前，如
    @fun_run_time()
    def xxx():
    ''' 
    @wraps(func)#可删去，是用来显示原始函数名的
    def _inner(*args, **kwargs):
        s_time = time.time()
        ret = func(*args, **kwargs)
        e_time = time.time()
        #
        print(colorstr("\t----function [{}] costs {} s".format(func.__name__, e_time-s_time), 'yellow'))
        return ret
    return _inner

def tic():
    '''
    开始计时。
    t = tic()
    '''
    s_time = time.time()
    return s_time


def toc(s_time, word='tic-toc', act_number = 1, mute=True):
    '''
    结束计时，返回毫秒数。
    t = toc(t, '模块函数名', '处理次数', True)\n
    mute代表不打印。
    '''
    e_time = time.time()
    temp = int((e_time-s_time)*1000)
    if not mute:
        if act_number > 1:
            print(colorstr(f"\t----module [{word}] costs {temp} ms, for {act_number} actions, ({int(temp/act_number)}ms/action)", 'yellow'))
        else:
            print(colorstr(f"\t----module [{word}] costs {temp} ms", 'yellow'))
    return temp