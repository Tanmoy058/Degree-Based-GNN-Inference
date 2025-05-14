import torch
import dgl
import ogb
import time
time.sleep(2)
from ogb.nodeproppred import DglNodePropPredDataset
import gc
import numpy as np
from dgl.nn import SAGEConv
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
from tqdm import trange
import random
import torch.multiprocessing as mp
import threading
import math
import socket
import pickle
from dgl.convert import create_block
from torch import cat
from dgl.transforms import to_block
import math
#import utils
HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server
WIDTH = 8
LAST = 9999999999
NTYPE = "_TYPE"
NID = "_ID"
ETYPE = "_TYPE"
EID = "_ID"
TIMEOUT = 300



if __name__ == '__main__':
    mp.set_start_method('spawn')
    time = 50
    arrival_model = 'non-bursty'
    num_layers = 3
    GNN = 'SAGE'
    chunking = True
    sorts = [True]
    verbose = False
    num_trials = 5
    pipeline = False
    graph_name = 'friendster'
    file_name = 'pcts_FR_B.pkl'
    with open(file_name, 'rb') as f:
        pcts = pickle.load(f)
    bin_path = "dataset/friendster_dgl.bin"
    g_list, _ = dgl.load_graphs(bin_path)
    graph = g_list[0]
    del g_list
    gc.collect()
    #feats = torch.empty((graph.number_of_nodes(), 256), dtype = torch.float32)
    graph = graph.formats(formats = 'csc')
    gc.collect()
    
    
    max_id = graph.number_of_nodes() - 1
    rng = np.random.default_rng(seed = 42)
    input_queue = torch.zeros(100000, dtype = torch.int64)    
    input_queue[:] = torch.from_numpy(rng.choice(max_id, size = input_queue.shape[0], replace = True))

    
    graph.pin_memory_()
    load = torch.zeros(max_id + 1, dtype=torch.int64).to(device='cuda:0')
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    
    nids = input_queue.to(device='cuda:0')
    for i in trange(input_queue.shape[0]):
        blocks = sampler.sample_blocks(graph, nids[i])
        load[nids[i]] = blocks[2][0].srcdata[dgl.NID].shape[0]
        
    load = load.cpu()
    nids = nids.cpu()
    
    fn = 'FR_load.pkl'
    with open (fn, 'wb') as f:
        pickle.dump([nids, load], f) 
    
        
            
       

            
            


                                         
