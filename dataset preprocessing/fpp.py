import inspect
import json
import os
import pickle
import shutil
import time
import zipfile
from functools import partial, reduce, wraps
from timeit import default_timer

import dgl

import numpy as np
import pandas
import requests
import torch





def get_friendster():
    df = pandas.read_csv(
        "com-friendster.ungraph.txt.gz",
        sep="\t",
        skiprows=4,
        header=None,
        names=["src", "dst"],
        compression="gzip",
    )
    src = df["src"].values
    dst = df["dst"].values
    print("construct the graph")
    return dgl.graph((src, dst))

format = ["csc"]    
bin_path = "friendster_dgl.bin"    
g = dgl.compact_graphs(get_friendster()).formats(format)
dgl.save_graphs(bin_path, [g])
