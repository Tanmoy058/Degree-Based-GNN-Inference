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
import utils
HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server
WIDTH = 8
LAST = 9999999999
NTYPE = "_TYPE"
NID = "_ID"
ETYPE = "_TYPE"
EID = "_ID"
TIMEOUT = 100

if __name__ == '__main__':
    graph_name = 'ogbn-papers100M'
    dataset = DglNodePropPredDataset(graph_name)
    graph, node_labels = dataset[0]
    node_labels = torch.nan_to_num(node_labels)
    feats = graph.ndata['feat']
    max_id = graph.number_of_nodes() - 1
    graph.ndata.pop('feat')
    file_name = 'feats.pkl'
    del feats
    gc.collect()
    print('creating formats ...')
    (u, v) = graph.edges('uv')
    U = torch.empty(u.shape[0]*2, dtype = u.dtype)
    V = torch.empty(u.shape[0]*2, dtype = u.dtype)
    U[0:u.shape[0]] = u
    U[u.shape[0]:] = v
    V[0:u.shape[0]] = v
    V[u.shape[0]:] = u
    del dataset, u, v, graph
    gc.collect()
    graph_c = dgl.graph((U, V))
    graph_c.create_formats_()
    gc.collect()
    print('format created') 
    file_name = 'PA.pkl'
    dataset = DglNodePropPredDataset(graph_name)
    graph, node_labels = dataset[0]
    feats = graph.ndata['feat']
    data = [graph_c, feats]
    with open(file_name, 'wb') as f:
            pickle.dump(data, f)
