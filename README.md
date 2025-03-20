# PipeTGL

This repository contains python codes for this paper:
> PipeTGL: (Near) Zero Bubble Memory-based Temporal Graph Neural Network Training via Pipeline Optimization

## 📚 Introduction
In this paper, we propose a pipeline parallel approach for multi-GPU M-TGNN training that effectively addresses both inter-minibatch memory dependencies and intra-minibatch task dependencies, based on a runtime analysis DAG for M-TGNNs. We further optimize pipeline efficiency by incorporating improved scheduling, finer-grained operation reorganization, and targeted communication optimizations tailored to the specific training properties of M-TGNN. These enhancements significantly reduce GPU waiting and idle time caused by memory dependencies and frequent communication and result in zero pipeline bubbles for common training configurations. Extensive evaluations demonstrate that PipeTGL achieves a speedup of 1.27x to 4.74x over other baselines while also improving the accuracy of M-TGNN training across multiple GPUs.

## 🔑 Setup

### 1.Dependencies

Our development environment:
- Ubuntu 22.04
- g++ 9.4
- CUDA 12.4
- cmake 3.23
- torch >= 1.10
- dgl >= 1.1.1 (CUDA version)
- tqdm >= 4.66.2
- scikit-learn >= 1.3.0
- pandas >= 2.2.1
- numpy >= 1.26.4

### 2.Clone repository

```sh
git clone https://github.com/magicmoo/PipeTGL.git
```

### 3.Install GNNFlow
We implemented PipeTGL based on GNNFlow. Before you start PipeTGL, you should install GNNFlow in your python environment.
```sh
git clone https://github.com/jasperzhong/GNNFlow.git
python setup.py install
```

## 🗃️ Datasets

### Download the datasets
```sh
cd scripts/ && ./download_data.sh
```

## 🚀 Get Started

### 1.Multi-GPU single machine
Training TGN model on the \<DatasetName> dataset with \<NumberOfEpochs> epochs on \<NumberOfGpus> GPUs within a single mechine.
```sh
./scripts/run_single_mechine.sh <NumberOfEpochs> <DatasetName> <NumberOfGpus>


# Training TGN model on the REDDIT dataset with 50 epochs on 4 GPUs within a single mechine:
./scripts/run_single_mechine.sh 50 REDDIT 4   
```

### 2.Distributed training
Training TGN model on the \<DatasetName> dataset with \<NumberOfEpochs> epochs on \<NumberOfGpus> GPUs per machine across a total number of \<NumberOfNodes> machines.

```sh
./scripts/run_multi_mechines.sh <NumberOfEpochs> <DatasetName> <NumberOfGpus> <NumberOfNodes>


# Training TGN model on the GDELT dataset with 100 epochs on 4 GPUs per machine across a total number of 2 machines:
./scripts/run_multi_mechines.sh 100 GDELT 4 2   
```