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
TIMEOUT = 150




def client(max_id, number_of_requests, arrival_model, arrival_rate, HOST, PORT, WIDTH, LAST, verbose):
    CVs = np.array([1, 1])
    transition_ps = np.array([0.01, 0.01])
    if arrival_model == 'bursty':
        request_rates = np.array([1, 2])
        K = 2/(1/request_rates[0] + 1/request_rates[1])
        request_rates = request_rates * arrival_rate / K
    else:
        request_rates = np.array([arrival_rate, arrival_rate])


    alphas = (1/CVs)**2
    betas = alphas * request_rates
    thetas = 1/betas
    rt = 1/((transition_ps[1]/request_rates[0] + transition_ps[0]/request_rates[1])/(transition_ps[1] + transition_ps[0]))
    rng = np.random.default_rng(seed = 42)
    if verbose:
        print('Client: Generating the trace ...')
    input_queue = np.zeros(1000000, dtype = np.int64)    
    input_queue[:] = rng.choice(max_id, size = input_queue.shape[0], replace = True)
    input_queue = input_queue[0:number_of_requests]
    request_intervals = np.zeros((CVs.shape[0], number_of_requests))
    for i in range(CVs.shape[0]):
        request_intervals[i,0:number_of_requests] = rng.gamma(shape = alphas[i], scale = thetas[i], size = 10000000)[0:number_of_requests]
    rnd = rng.uniform(size = 10000000)[0:number_of_requests]
    burst_state = False
    change_state = False
    request_times = np.zeros((number_of_requests))
    pointers = np.zeros(CVs.shape[0], dtype = np.int64)
    for i in range(1, number_of_requests):
        if not(burst_state):
            interval = request_intervals[0, pointers[0]]
            change_state = rnd[i] < transition_ps[0]
            pointers[0] += 1
        else:
            interval = request_intervals[1, pointers[1]]
            change_state = rnd[i] < transition_ps[1]
            pointers[1] += 1
        request_times[i] = request_times[i-1] + interval
        burst_state = burst_state^change_state
    if verbose:
        print('Client: Trace generated.')



    if verbose:
        print('Client: Starting the client ...')
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if verbose:
        print('Client: Client started. Waiting for the server ...')
    t_w = 0
    cn = False
    while True:
        try:
            s.connect((HOST, PORT))
            if verbose:
                print('Client: Connected to the server. Sending inference requests ...')
            cn = True
            break
        except:
            time.sleep(1)
            t_w += 1
            if t_w > 150:
                print('Client: Server timed out.')
                break



    if cn:
        i = 0
        t_ref = time.time()
        while (i < number_of_requests):
            if request_times[i] < (time.time() - t_ref):
                message = int(input_queue[i]).to_bytes(WIDTH, 'little', signed = False)
                s.sendall(message)
                i += 1
            else:
               time.sleep(0.0001)
        #s.shutdown()
        message = int(LAST).to_bytes(WIDTH, 'little', signed = False)
        s.sendall(message)
        time.sleep(5)
        s.close()
        if verbose:
            print('Client: ', time.time() - t_ref)
    else:
        print('Client: failure.')





def dim1(tensor):
    tensor = torch.squeeze(tensor)
    device = tensor.device
    if tensor.dim() == 0 and tensor.numel() == 1:
        tensor = torch.tensor([tensor.cpu().item()]).to(device = device)
    return tensor
    
    
def cache_inds(ins, cache_meta):
    ins = dim1(ins)
    tmp = torch.squeeze(cache_meta[ins])
    if tmp.dim() == 0:
       tmp = torch.tensor([tmp.cpu().item()]).to(device = 'cuda:0')
    plc1 = dim1(torch.squeeze(torch.nonzero(tmp>=0)))
    plc1_g = dim1(torch.squeeze(tmp[plc1]))
    plc2 = dim1(torch.squeeze(torch.nonzero(tmp<0)))
    plc2_c = dim1(torch.squeeze(ins[plc2]))
    return (ins, plc1, plc1_g, plc2, plc2_c)

def rcv(input_queue, arrival_times, arrival_order, input_pointers, num_requests, WIDTH, LAST, conn):
    while True:
        data = conn.recv(WIDTH)
        d = int.from_bytes(data, 'little', signed = False)
        if d == LAST:
            num_requests[0] = input_pointers[0]
            break
        input_queue[input_pointers[0]] = int.from_bytes(data, 'little', signed = False)
        arrival_times[input_pointers[0]] = time.perf_counter_ns()
        arrival_order[input_pointers[0]] = input_pointers[0]
        input_pointers[0] += 1

def rspnd(res, input_pointers, arrival_order, num_requests, conn):
    p = 0
    while True:
        if p<input_pointers[2]:
            message = "{}: {}".format(str(arrival_order[p].item()), str(res[p].item())).encode('utf-8')
            try:
                conn.sendall(message)
            except:
                break
            p += 1
        if p >= num_requests[0] - 1:
            break
        time.sleep(0.01)

def request_handler(input_queue, arrival_times, input_pointers, HOST, PORT, WIDTH, LAST, barrier, infer_en, num_requests, arrival_order, res, verbose):
    if verbose:
        print('Server: request handler started.')
    barrier.wait()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen()
    if verbose:
        print('Server: server ready.')
    conn, addr = s.accept()
    rcv_thread = threading.Thread(target = rcv, args = (input_queue, arrival_times, arrival_order, input_pointers, num_requests, WIDTH, LAST, conn))
    rspnd_thread = threading.Thread(target = rspnd, args = (res, input_pointers, arrival_order, num_requests, conn))
    if verbose:
        print('Server: Connected to the client.')
    rcv_thread.start()
    rspnd_thread.start()
    rcv_thread.join()
    rspnd_thread.join()
    conn.close()
    s.close()


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))

    def forward(self, bipartites, x):
        n_layers = self.n_layers
        
        #first layer
        blks = bipartites[0]
        num_chunks = len(blks) - 1
        inds = blks[-1]
        #print(num_chunks)
        for i in range(num_chunks):
            if i == 0:
                x_i = self.layers[0](blks[i][0], x[blks[i][1]])
            else:
                torch.cuda.empty_cache()
                x_i[inds[i]: inds[i+1]] = self.layers[0](blks[i][0], x[blks[i][1]])[inds[i]: inds[i+1]]
        x = F.relu(x_i)
        
        #mid layers
        for l in range(1, n_layers - 1):
            x = self.layers[l](bipartites[l], x)
            x = F.relu(x)
        
        #last layer
        x = self.layers[n_layers - 1](bipartites[n_layers - 1], x)
        return x
        
        
        
class GAT2(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, num_heads):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv(in_feats, n_hidden, num_heads, allow_zero_in_degree = True))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GATConv(n_hidden, n_hidden, num_heads, allow_zero_in_degree = True))
        self.layers.append(dglnn.GATConv(n_hidden, n_classes, num_heads, allow_zero_in_degree = True))

    def forward(self, bipartites, x, op = 0):
        n_layers = self.n_layers
        if op == 0:
            
            x = self.layers[0](bipartites[0], x)
            x = F.relu(x)


            #mid layers
            for l in range(1, n_layers - 1):
                x = self.layers[l](bipartites[l], x)
                x = F.relu(x)

            #last layer
            x = self.layers[n_layers - 1](bipartites[n_layers - 1], x)
        elif op == 1:
            x = self.layers[0](bipartites, x)
            x = F.relu(x)
        elif op == 2:
            #mid layers
            for l in range(1, n_layers - 1):
                x = self.layers[l](bipartites[1 + len(bipartites) - n_layers], x)
                x = F.relu(x)
            #last layer
            x = self.layers[n_layers - 1](bipartites[-1], x)
        else:
            raise ValueError("invalid operation")                    
        return x

class SAGE2(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
    def forward(self, bipartites, x, op = 0):
        n_layers = self.n_layers
        if op == 0:
            
            x = self.layers[0](bipartites[0], x)
            x = F.relu(x)


            #mid layers
            for l in range(1, n_layers - 1):
                x = self.layers[l](bipartites[l], x)
                x = F.relu(x)

            #last layer
            x = self.layers[n_layers - 1](bipartites[n_layers - 1], x)
        elif op == 1:
            x = self.layers[0](bipartites, x)
            x = F.relu(x)
        elif op == 2:
            #mid layers
            for l in range(1, n_layers - 1):
                x = self.layers[l](bipartites[1 + len(bipartites) - n_layers], x)
                x = F.relu(x)
            #last layer
            x = self.layers[n_layers - 1](bipartites[-1], x)
        else:
            raise ValueError("invalid operation")                    
        return x

class GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, num_heads):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv(in_feats, n_hidden, num_heads, allow_zero_in_degree = True))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GATConv(n_hidden, n_hidden, num_heads, allow_zero_in_degree = True))
        self.layers.append(dglnn.GATConv(n_hidden, n_classes, num_heads, allow_zero_in_degree = True))

    def forward(self, bipartites, x, op = 0):
        n_layers = self.n_layers
        if op == 0:
            
            x = self.layers[0](bipartites[0], x)
            x = F.relu(x)


            #mid layers
            for l in range(1, n_layers - 1):
                x = self.layers[l](bipartites[l], x)
                x = F.relu(x)

            #last layer
            x = self.layers[n_layers - 1](bipartites[n_layers - 1], x)
        elif op == 1:
            x = self.layers[0](bipartites, x)
            x = F.relu(x)
        elif op == 2:
            #mid layers
            for l in range(1, n_layers - 1):
                x = self.layers[l](bipartites[1 + len(bipartites) - n_layers], x)
                x = F.relu(x)
            #last layer
            x = self.layers[n_layers - 1](bipartites[-1], x)
        else:
            raise ValueError("invalid operation")                    
        return x
        
        
        
        
        
        
class MySampler():

    def __init__(
        self,
        num_layers,
        graph,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
        fused=True,
        chunk_size = 30000,
        compact_threshold = 2000000,
    ):
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.num_layers = num_layers
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.g = None
        self.output_device = output_device
        self.chunk_size = chunk_size
        num_nodes = graph.number_of_nodes()
        self.hash_ind = torch.zeros(num_nodes, device = 'cuda:0', dtype = torch.bool)
        self.map = torch.zeros(num_nodes, device = 'cuda:0', dtype = torch.long)
        self.compact_threshold = compact_threshold
        self.t_c = 0
        mtx = graph.adj_tensors('csc')
        U = mtx[1]
        indptr = mtx[0]
        indptr_gpu = indptr.to(device = 'cuda:0')
        if not(U.is_pinned()):
            dgl.utils.pin_memory_inplace(U)
        self.U = U
        #self.in_degs = graph.in_degrees().to(device = 'cuda:0')
        self.indptr_gpu = indptr_gpu
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for l in range(self.num_layers - 1):
            frontier = g.sample_neighbors(
                seed_nodes,
                -1,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            block = self.compact(frontier, seed_nodes)
            # If sampled from graphbolt-backed DistGraph, `EID` may not be in
            # the block. If not exists, we should remove it from the block.
            if EID in frontier.edata.keys():
                block.edata[EID] = frontier.edata[EID]
            else:
                del block.edata[EID]
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)
        

        #indptr_out = torch.empty(seed_nodes.shape[0] + 1, device = seed_nodes.device, dtype = self.in_degs.dtype)
        #torch.cumsum(self.in_degs[seed_nodes], dim = 0, out = indptr_out[1:])
        #indptr_out[0] = 0
        #frontier = g.sample_neighbors(
        #    seed_nodes,
        #    -1,
        #    edge_dir=self.edge_dir,
        #    prob=self.prob,
        #    replace=self.replace,
        #    output_device=self.output_device,
        #    exclude_edges=exclude_eids,
        #)
        [u, v, indptr_out] = dgl.utils.Sample(self.U, self.indptr_gpu, seed_nodes, num_threads = torch.tensor([4096], dtype = torch.long))
        #(u,v) = frontier.edges()
        num_chunks = max(min(math.ceil(u.shape[0] / self.chunk_size), 10), 1)
        chunk_size = (seed_nodes.shape[0] // num_chunks + 1)
        bounds = [min(i*chunk_size, seed_nodes.shape[0]) for i in range(num_chunks + 1)]
        for i in range(num_chunks):
            seeds = seed_nodes[bounds[i]:bounds[i+1]]
            #frontier = g.sample_neighbors(
            #    seeds,
            #    -1,
            #    edge_dir=self.edge_dir,
            #    prob=self.prob,
            #    replace=self.replace,
            #    output_device=self.output_device,
            #    exclude_edges=exclude_eids,
            #)
            #[u, v, indptr] = dgl.utils.Sample(self.U, self.indptr_gpu, seeds, num_threads = torch.tensor([4096], dtype = torch.long))
            #frontier = dgl.graph((u_c, v_c))
            u_c = u[indptr_out[bounds[i]]:indptr_out[bounds[i+1]]]
            v_c = v[indptr_out[bounds[i]]:indptr_out[bounds[i+1]]]
            block = self.compact((u_c, v_c), seeds, uv=True)
            blocks.insert(0, block)
        return [seed_nodes, output_nodes, blocks]
    def compact(self, frontier, seeds, uv = False):
      torch.cuda.synchronize()
      t0 = time.time()
      if uv or frontier.number_of_edges() > self.compact_threshold:
        if uv:
            (u, v) = frontier
        else:
            (u,v) = frontier.edges('uv')
        self.hash_ind[u] = True
        self.hash_ind[seeds] = False
        map_pos = torch.squeeze(torch.cat((dim1(seeds), dim1(torch.squeeze(torch.nonzero(self.hash_ind)))), dim = 0))
        self.map[map_pos] = torch.arange(map_pos.shape[0], device = 'cuda:0')
        u = self.map[u]
        v = self.map[v]
        block = create_block((u,v), num_src_nodes = map_pos.shape[0], num_dst_nodes = seeds.shape[0])
        block.srcdata[NID] = map_pos
        block.dstdata[NID] = seeds
        self.hash_ind[map_pos] = False
      else:
        block = to_block(frontier, seeds)
      torch.cuda.synchronize()
      t1 = time.time()
      self.t_c += t1 - t0
      return block
        
        
        
        

def fetch(feats, indices, f0, f1, barrier, bs, be, infer_en):
    if not(feats.is_pinned()):
        cudart = torch.cuda.cudart()
        rt = cudart.cudaHostRegister(feats.data_ptr(), feats.numel() * feats.element_size(), 0)
    barrier.wait()
    while infer_en[0]:
        bs.wait()
        sz = indices[0].cpu().item()
        adr = indices[1].cpu().item()
        idx = indices[2:2+sz]
        if adr:
            f1[0:sz] = dgl.utils.gather_pinned_tensor_rows(feats, idx)
        else:
            f0[0:sz] = dgl.utils.gather_pinned_tensor_rows(feats, idx)
        be.wait()
        




def inference(feats_file_name, input_queue, graph, pointers, infer_en, arrival_times_actual, num_requests, barrier, arrival_order, res, in_degrees, feats, num_layers, opt, sort, chunking, cache_size, GNN, verbose, nsz, max_mem):
    if verbose:
        print('Server: inference process started.')
    device = 'cuda:0'
    Ts = 0
    Ti = 0
    Tt = 0
    mem_usage = []
    Time = []
    if chunking:
        #chunk_size = 25000
        chunk_size = opt * (torch.mean(in_degrees.to(dtype = torch.float)).cpu().item())**(num_layers - 2) * 10
        chunk_size = chunking
        #print(chunk_size)
        
    else:
        chunk_size = 1000000000000
    #print('loading features ...')
    #with open(feats_file_name, 'rb') as f:
        #feats = pickle.load(f)
    if GNN == 'SAGE':
        model = SAGE2(feats.shape[1],256, 100, num_layers).to(device = 'cuda:0')
    elif GNN == 'GAT':
        model = GAT2(feats.shape[1],256, 47, num_layers, 1).to(device = 'cuda:0')
    model.eval()
    if not(feats.is_pinned()):
        cudart = torch.cuda.cudart()
        rt = cudart.cudaHostRegister(feats.data_ptr(), feats.numel() * feats.element_size(), 0)
    if not(graph.is_pinned()):
        graph.unpin_memory_()
        graph.pin_memory_()
    #sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    #sampler = Sampler(num_layers, graph, 'cuda:0', chunk_size = chunk_size)
    sampler = MySampler(num_layers, graph, chunk_size = chunk_size, compact_threshold = 100000)
    #cache_size = int((cache_size_GB * 1000000000) / feats.element_size() / feats.shape[1])
    #cache_size = 35000000
    cache_size = min(cache_size, in_degrees.shape[0])
    if verbose:
        print('Server: opt, chunk size, cache size ', opt, chunk_size, cache_size)
    in_degs = graph.in_degrees()
    cached_feats = torch.squeeze(torch.sort(in_degs, descending = True)[1][0:cache_size])
    torch.cuda.synchronize()
    _cache_ = feats[cached_feats].to(device = 'cuda:0')
    torch.cuda.synchronize()
    cache_meta = torch.zeros(feats.shape[0], device = 'cuda:0', dtype = torch.long) - 1
    uniq = torch.zeros(feats.shape[0], device = 'cuda:0', dtype = torch.bool)
    torch.cuda.synchronize()
    cache_meta[cached_feats] = torch.from_numpy(np.arange(cache_size, dtype = np.int64)).to(device = 'cuda:0')
    torch.cuda.synchronize()
    response_times = torch.zeros(input_queue.shape[0], dtype = torch.long)
    err = 0
    e = 0
    pr = 0
    hr = torch.zeros(2, dtype = torch.long) + 1
    failure = False
    n1 = 0
    n2 = 0
    if chunk_size < 1000000000:
        input_features = torch.empty((max(int(chunk_size/3.5)+ 5000000, 5000000), feats.shape[1]), dtype = feats.dtype, device = 'cuda:0')
    else:
        input_features = torch.empty((15000000, feats.shape[1]), dtype = feats.dtype, device = 'cuda:0')    
    torch.cuda.synchronize()
    barrier.wait()
    T_st = time.time()    
    try:
        with torch.no_grad():
            while infer_en[0]:
                while infer_en[0]:
                    if pointers[0]>pointers[1]:
                        sched(pointers, in_degrees, input_queue, arrival_order, arrival_times_actual, opt, sort)
                        break
                    if pointers[2] >= (num_requests[0]):
                        infer_en[0] = False
                    time.sleep(0.002)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                m_max = torch.cuda.max_memory_allocated()
                mem_usage.append(m_max)
                torch.cuda.reset_peak_memory_stats()
                
                Time.append(time.time() - T_st)
                p = pointers[2].item()
                bs = pointers[1].item() - p
                    #print('i', p, bs, pointers)
                seeds = input_queue[p:p + bs]
                seeds = seeds.clone().to(device = 'cuda:0')
                try:
                    minibatch = sampler.sample_blocks(graph, seeds)
                    cnt = True
                    X = []
                    n1 += 1
                    lns = []
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    Ts += t1 - t0
                    for l in range(len(minibatch[2]) - num_layers + 1):
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        if torch.cuda.memory_allocated() > 1024*1024*1024*38:
                            gc.collect()
                            torch.cuda.empty_cache()
                        n2 += 1
                        block = minibatch[2][l]
                        ins = block.srcdata["_ID"]
                        lns.append(ins.shape[0])
                        if ins.shape[0] > input_features.shape[0]:
                            cnt = False
                            break
                        (ins, plc1, plc1_g, plc2, plc2_c) = cache_inds(ins, cache_meta)
                        if plc1.numel():
                            dgl.utils.index_select_from_to(_cache_, input_features, plc1_g, plc1)
                        if plc2.numel():
                            input_features[plc2] = dgl.utils.gather_pinned_tensor_rows(feats, plc2_c.to(device = 'cuda:0'))
                        torch.cuda.synchronize()
                        t2 = time.perf_counter()
                        Tt += t2 - t1
                        #input_features = dgl.utils.gather_pinned_tensor_rows(feats, ins.to(device = 'cuda:0'))
                        outs = model(block, input_features[0:ins.shape[0]], op = 1)
                        X.append(outs)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        t3 = time.perf_counter()
                        Ti += t3 - t2
                    if cnt:
                        #if len(lns) > 1:
                            #print(lns)
                        torch.cuda.synchronize()
                        t2 = time.perf_counter()
                        X = torch.cat(X, dim = 0)
                        outs = model(minibatch[2], X, op = 2)
                        torch.cuda.synchronize()
                        t3 = time.perf_counter()
                        Ti += t3 - t2
                    else:
                        e+= bs
                    pr+=bs
                except:
                    e+=bs
                response_times[p: p + bs] = time.perf_counter_ns()
                pointers[2] = p + bs
                #print(p+bs)
                if verbose:
                    print(pointers[2].cpu().item(), num_requests[0].item(), err)
                if pointers[2] >= (num_requests[0]):
                    infer_en[0] = False
    except Exception as error_desc:
    #else:
        #print(error_desc)
        print('failure ', pointers[2], bs, ins.shape[0], error_desc)
        failure = True
    #print(Ts, Tt, Ti)
    l = num_requests[0]
    l -= 1
    response_times = response_times[0:l]
    arrival_times_actual = arrival_times_actual[0:l]
        #dist = torch.bitwise_right_shift(response_times - arrival_times_actual, 20).detach().numpy()
        #print(np.sum(dist<0))
        #for i in range(20):
        #    print(i*5, np.sort(dist)[int(dist.shape[0]*(i*0.05))])
        #print(99, np.sort(dist)[int(dist.shape[0]*(0.99))])
        #if chunk_size < 50000000:
        #    file_name = 'dist_proposed.pkl'
        #else:
        #    file_name = 'dist_bl.pkl'
        #with open (file_name, 'wb') as f:
        #    pickle.dump(dist, f)
    mem_usage.sort()
        #print('max memory usage = ', mem_usage[int(len(mem_usage)*0.995)])
    print('Server: average latency = ', int(np.mean(torch.bitwise_right_shift(response_times - arrival_times_actual, 20).detach().numpy())), '(ms) failures =', e, [Ts, Tt, Ti], n1, n2, sampler.t_c)
    if e>=l/100:
        failure = True
    if not(failure):
        max_mem[0] = mem_usage[int(len(mem_usage)) - 2]
    #print(hr, hr[1].cpu().item()/hr[0].cpu().item())
    




        

    
def sched(pointers, in_degrees, input_queue, arrival_order, arrival_times_actual, opt, sort):
    adapt = True
    alpha = 0.00000001
    max_window = 128
    max_bs = 64
    window = min(max_window, pointers[0] - pointers[1])
    seeds = input_queue[pointers[1]:pointers[1]+window]
    degs = in_degrees[seeds]
    T_arr = (arrival_times_actual[pointers[1]: pointers[1] + window])
    t_promotion = 40000_000_000
    if True:
        if False:
            #dec = degs - torch.pow(alpha * (time.perf_counter_ns() - T_arr), 1)
            #print(torch.sum(((time.perf_counter_ns() - T_arr) > t_promotion)))
            dec = degs - ((time.perf_counter_ns() - T_arr) > t_promotion) * (1024*1024*1024*1024)
            
        else:
            dec = T_arr
        [_, indices] = torch.sort(dec)
        seeds = seeds[indices]
        input_queue[pointers[1]: pointers[1] + window] = seeds
        degs_q = degs[indices]
        arr = arrival_times_actual[pointers[1]: pointers[1] + window]
        arr = arr[indices]
        arrival_times_actual[pointers[1]: pointers[1] + window] = arr
        o = arrival_order[indices + pointers[1]]
        arrival_order[pointers[1]: pointers[1] + window] = o
    else:
        degs_q = degs
    if adapt:
        p = 0
        deg_sum = 0
        step_size = 8
        for step in range(degs_q.shape[0]):
            deg_sum += degs_q[step]
            if deg_sum >= opt:
                break
            p += 1
        p = max(min(degs_q.shape[0], p), 1)
    else:
        p = max(min(degs_q.shape[0], max_bs), 1)
    pointers[1] = pointers[1] + p




def main(number_of_requests, arrival_model, arrival_rate, num_layers, opt, sort, chunking, cache_size, GNN, pipeline, verbose, graph, feats, nsz, max_mem):
    use_mps = pipeline
    graph_name = 'ogbn-papers100M'
    max_mem.share_memory_()
    pct = [70]
    if verbose:
        print('\n', arrival_model, arrival_rate, num_layers, opt, sort, chunking, GNN)
    if use_mps:
        user_id = utils.mps_get_user_id()
        utils.mps_daemon_start()
        utils.mps_server_start(user_id)
        server_pid = utils.mps_get_server_pid()
        time.sleep(10)
        utils.mps_set_active_thread_percentage(server_pid, pct[0])
        time.sleep(10)
        if verbose:
            print('using mps, active thread percentage set to ', pct[0])
    #graph_name = 'ogbn-arxiv'
    file_name = 'PA.pkl'
    #with open(file_name, 'rb') as f:
    #    [graph_a, feats] = pickle.load(f)
    #graph = graph_a.formats(formats = 'csc')
    #del graph_a
    #gc.collect()
    #if verbose:
        #print('creating formats ...')
    #graph.create_formats_()
    num_nodes = graph.number_of_nodes()
    in_degrees = graph.in_degrees()
    max_id = in_degrees.shape[0] - 1
    #feats.share_memory_()
    queue_length = 20000000











    num_requests = torch.zeros(1, dtype = torch.long)
    num_requests[0] = 4000000000
    num_requests.share_memory_()



    input_queue = torch.zeros(queue_length, dtype = torch.long)
    input_queue.share_memory_()
    pointers = torch.zeros(3, dtype = torch.long)
    pointers.share_memory_()
    infer_en = torch.zeros(1, dtype = torch.bool)
    infer_en[0] = True
    infer_en.share_memory_()
    arrival_order = torch.zeros(queue_length, dtype = torch.long)
    arrival_order.share_memory_()
    res = torch.zeros(queue_length, dtype = torch.long)
    res.share_memory_()




    arrival_times_actual = torch.zeros(input_queue.shape[0], dtype = torch.long)
    arrival_times_actual.share_memory_()
    execution_queue = []
    if pipeline:
        barrier = mp.Barrier(3)
    else:
        barrier = mp.Barrier(2)
    if verbose:
        print('spawning processes ...')
    request_handler_process = mp.Process(target = request_handler, args = (input_queue, arrival_times_actual, pointers, HOST, PORT, WIDTH, LAST, barrier, infer_en, num_requests, arrival_order, res, verbose))
    if pipeline:
        inference_process = mp.Process(target = inference_pipeline, args = (file_name, input_queue, graph, pointers, infer_en, arrival_times_actual, num_requests, barrier, arrival_order, res, in_degrees, feats, num_layers, opt, sort, chunking, cache_size, GNN, verbose))
    else:
        inference_process = mp.Process(target = inference, args = (file_name, input_queue, graph, pointers, infer_en, arrival_times_actual, num_requests, barrier, arrival_order, res, in_degrees, feats, num_layers, opt, sort, chunking, cache_size, GNN, verbose, nsz, max_mem))       
    client_process = mp.Process(target = client, args = (max_id, number_of_requests, arrival_model, arrival_rate, HOST, PORT, WIDTH, LAST, verbose))



    client_process.start()
    request_handler_process.start()
    inference_process.start()
    client_process.join()
    t_ws = time.time()
    while True:
        if ((request_handler_process.is_alive() or inference_process.is_alive()) and ((time.time() - t_ws) > TIMEOUT)):
            try:
                request_handler_process.terminate()
                #inference_process.terminate()
            except:
                a = 1
            try:
                #request_handler_process.terminate()
                inference_process.terminate()
            except:
                a = 1
        elif not(request_handler_process.is_alive()) and not(inference_process.is_alive()):
            break
        else:
            time.sleep(3)
    request_handler_process.join()
    inference_process.join()
    if use_mps:
        time.sleep(10)
        utils.mps_quit()




def run(number_of_requests, arrival_model, arrival_rate, num_layers, opt, sort, chunking, cache_size, GNN, pipeline, verbose, num_trials):
    for _ in range(num_trials):
        p = mp.Process(target = main, args = (number_of_requests, arrival_model, arrival_rate, num_layers, opt, sort, chunking, cache_size, GNN, pipeline, verbose, graph, feats))
        p.start()
        p.join()

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
    graph_name = 'ogbn-papers100M'
    file_name = 'pcts_PA_P.pkl'
    with open(file_name, 'rb') as f:
        pcts = pickle.load(f)
    file_name = 'dataset/PA.pkl'
    with open(file_name, 'rb') as f:
        [graph_a, feats] = pickle.load(f)
    graph = graph_a.formats(formats = 'csc')
    del graph_a
    gc.collect()
    
    number_of_requests = 5000
    opts = list(pcts.keys())
    cnds = torch.zeros((2,30, 3), dtype = torch.long)
    max_mem = torch.zeros(1, dtype = torch.long)
    max_mem.share_memory_()
    base_config = [150, 200, 1000000, False, 5000000, 10500000]
    j = 0
    for i in range(1,4):
        for k in range(10):
            pct_tar = 0.75 + 0.02*k
            max_mem[0] = 999999999999
            config = [base_config[i] for i in range(len(base_config))]
            opt = opts[i]
            pcts_i = pcts[opt]
            chunk_size = int(pcts_i[int(pct_tar*int(pcts_i.shape[0]))])
            config[1] = opt
            config[4] = chunk_size
            print(config)
            try:
                p = mp.Process(target = main, args = (number_of_requests, arrival_model, config[0], num_layers, config[1], config[3], config[4], config[2], GNN, pipeline, verbose, graph, feats, config[5], max_mem))
                p.start()
                p.join()
            except:
                print(fail)
            print('max memory usage = ', max_mem[0])
            cnds[0,j,0] = opt
            cnds[0,j,1] = chunk_size
            cnds[0,j,2] = max_mem[0]
            j += 1
    with open ('sens_PA_P.pkl', 'wb') as f:
        pickle.dump(cnds, f)         
        
    GNN = 'GAT'
    j = 0
    base_config = [35, 200, 1000000, False, 5000000, 10500000]
    for i in range(0,3):
        for k in range(10):
            pct_tar = 0.96 + 0.0030*k
            max_mem[0] = 999999999999
            config = [base_config[i] for i in range(len(base_config))]
            opt = opts[i]
            pcts_i = pcts[opt]
            chunk_size = int(pcts_i[int(pct_tar*int(pcts_i.shape[0]))])
            config[1] = opt
            config[4] = chunk_size
            print(config)
            try:
                p = mp.Process(target = main, args = (number_of_requests, arrival_model, config[0], num_layers, config[1], config[3], config[4], config[2], GNN, pipeline, verbose, graph, feats, config[5], max_mem))
                p.start()
                p.join()
            except:
                print(fail)
            print('max memory usage = ', max_mem[0])
            cnds[1,j,0] = opt
            cnds[1,j,1] = chunk_size
            cnds[1,j,2] = max_mem[0]
            j += 1
    with open ('sens_PA_P.pkl', 'wb') as f:
        pickle.dump(cnds, f)         
        
        
            
       

            
       

            
            


                                         
