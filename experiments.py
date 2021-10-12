import PyTorchDiscreteFlows.discrete_flows.disc_utils as disc_utils
from PyTorchDiscreteFlows.discrete_flows.made import MADE
from PyTorchDiscreteFlows.discrete_flows.mlp import MLP
from PyTorchDiscreteFlows.discrete_flows.embed import EmbeddingLayer
from PyTorchDiscreteFlows.discrete_flows.disc_models import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
from scipy.stats import multivariate_normal
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import itertools
import math
import random
import time
import os
from preprocessing import *
from train_utils import *

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(seed)
DEBUG = False

plt.gray()
Train_MNIST(digit='all', disc_layer_type='bipartite', batch_size=500, epoch=100, hidden_layer=784, temp_decay=1, lr_decay=1, path='', save=True, image_process=True, CNN=False, test_per_epoch=True, dim=(14,14), sample_size=15, af='linear', alpha=1, beta=1, id_init=True)