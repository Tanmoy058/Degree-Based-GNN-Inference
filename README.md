# GSM: Graph Sampling and Memory Management for GNNs

This repository contains the implementation of GraphSage and GAT models with optimized graph sampling and memory management techniques for high-throughput inference on large graphs.

## Project Overview

GSM (Graph Sampling and Memory) is a system for efficient inference on large graphs using Graph Neural Networks (GNNs). It includes several key components:

- Optimized graph sampling techniques
- GPU memory management
- Caching strategies for feature retrieval
- Request scheduling for high-throughput inference
- Client-server architecture for handling inference requests

The system is designed to work with large-scale graphs such as Friendster and Papers100M, using both GraphSAGE and GAT (Graph Attention Networks) models.

## Repository Structure

- `e2e/`: End-to-end implementation of various models and experiments
- `ablation_study/`: Contains code for ablation studies on different components
- `DGL modifications/`: Custom CUDA extensions for DGL
- `dataset preprocessing/`: Tools for preprocessing graph datasets
- `experimental_analysis/`: Code for analyzing experimental results
- `sensitivity_analysis/`: Tools for sensitivity analysis of different parameters
- `overhead/`: Code for measuring performance overhead

## Key Features

- Custom neighbor sampling implementation with memory optimization
- Feature caching strategies based on node degrees
- Multi-process architecture for parallel processing
- Adaptive request scheduling based on graph structure
- Support for both GraphSAGE and GAT models

## Requirements

- Python 3.6+
- PyTorch
- DGL (Deep Graph Library)
- CUDA-capable GPU
- Libraries: numpy, ogb, tqdm

## Usage

### Running Experiments

```bash
# Run with specific parameters
python e2e/FR_P.py --dataset <dataset_name> --cache_ratio <ratio>

# Run with GPU
CUDA_VISIBLE_DEVICES=0 python e2e/FR_P.py
```

### Main Parameters

- `arrival_rate`: Inference request arrival rate
- `num_layers`: Number of GNN layers (default: 3)
- `GNN`: Model type ('SAGE' or 'GAT')
- `chunk_size`: Size of processing chunks for large graphs
- `cache_size`: Size of feature cache in GPU memory
- `opt`: Optimization parameter for scheduling

## Custom DGL Extensions

The repository includes custom CUDA extensions for DGL to optimize:

- Memory pinning
- Array operations
- Neighborhood sampling
- Index selection
