# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

import errno
import os
import yaml
import random
import torch
import numpy as np


def mkdir(path):
    # if it is the current folder, skip.
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as fp:
        return yaml.load(fp)


def find_file_path_in_yaml(fname, root):
    if fname is not None:
        if os.path.isfile(fname):
            return fname
        elif os.path.isfile(os.path.join(root, fname)):
            return os.path.join(root, fname)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), os.path.join(root, fname)
            )
