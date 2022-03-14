import os, sys
import pickle
import subprocess
import time

def colorstr(*input):
    '''
    输入一串字符和格式，用欧逗号隔开。默认是蓝色加粗。
    eg. 
        tempstr = 'as'

        print(colorstr(str(tempstr), 'blue'))

        print(colorstr('initializing experiment arguments...', 'blue'))

        temp = colorstr('initializing experiment arguments...', 'blue', 'underline')
    '''
    string, *args = input if len(input) > 1 else (input[0], 'blue', 'bold')  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']



def mkdir_p(path, delete=False, print_info=True):
    '''
    if path exist, and delete=True, then delete this path.
    if path not exist, then create the path.
    '''
    if path == '': return

    if delete:
        subprocess.call(('rm -r ' + path).split())
    if not os.path.exists(path):
        if print_info:
            print('mkdir -p  ' + path)
        subprocess.call(('mkdir -p ' + path).split())

        
def savepickle(data, file_path):
    '''
    save the data at file_path.
    if file_path not exit, create the path.
    e.g.
        savepickle(mat, "weights\\1.pickle")
    '''
    mkdir_p(os.path.dirname(file_path), delete=False)
    print('pickle into', file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def unpickle(file_path):
    '''
    load the picklefile, and return it.
    e.g.
        mat = unpickle("weights\\1.pickle")
    '''
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data