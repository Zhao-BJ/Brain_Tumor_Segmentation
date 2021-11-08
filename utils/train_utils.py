import logging
import datetime
import numpy as np
import torch
from torch.backends import cudnn


def reproducibility(cuda, seed):
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True                   # for faster GPU training when input size doesn`t vary


def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')       # Return a logger with specified name
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
