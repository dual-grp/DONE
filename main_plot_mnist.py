#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from utils.plot_utils import *
import torch
torch.manual_seed(0)
#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from algorithms.server.server import Server
from algorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)
    
numedges = 32
num_glob_iters = 100
dataset = "MNIST"

# defind parameters
local_epochs = [120,120,120,20,20,1]
learning_rate = [1,1,1,0.05,0.02,0.2]
eta =  [0.015,0.01,0.015,1,1,1]
eta0 = [1,1,1,1,1,1]
batch_size = [0,256,0,0,256,0]
algorithms = ["DONE","DONE", "Newton", "DANE", "FedDANE", "FirstOrder"]
L = [0,0,0,0,0,0,0,0,0,0,0.001,0]
plot_summary_mnist(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, eta = eta, eta0 = eta0, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)