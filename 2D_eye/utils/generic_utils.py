from __future__ import absolute_import, division, print_function, unicode_literals

import optparse
import os
import sys
import time
import pickle
import string
import numpy as np
import psutil
import random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def memory():
    '''
    Estimate CPU memory
    '''
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # Memory use in GB
    print('Memory usage in GBs:%.3f'%memoryUse)

def get_free_gpu():
    '''
    Determine the gpu number that is free
    '''
    tmpfile='tmp_%s'%id_generator()
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >./%s'%tmpfile)
    memory_available = [int(x.split()[2]) for x in open(tmpfile, 'r').readlines()]
    os.system('rm -rf ./%s'%tmpfile)
    return np.argmax(memory_available)

def parse_list(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(","))
