# PipeTGL

PipeTGL is a high throughput and (nearly) no bubble M-TGNN traning framework.

## Setup

### Setup CUDA

PipeTGL has been tested with CUDA 12.4 on RTX4090.

### Clone repository

```sh
git clone https://github.com/magicmoo/PipeTGL.git
```

### Install GNNFlow
```sh
git clone https://github.com/jasperzhong/GNNFlow.git
python setup.py install
```

## Datasets

### Download the datasets
```sh
cd scripts/ && ./download_data.sh
```

## Train

### Multi-GPU single machine
Training TGN model on the <DatasetName> dataset with <NumberOfEpochs> epochs on <NumberOfGpus> GPUs.
```sh
./scripts/run_single_mechine.sh <NumberOfEpochs> <DatasetName> <NumberOfGpus>
```

### Distributed training
Training TGN model on the <DatasetName> dataset with <NumberOfEpochs> epochs on <NumberOfGpus> GPUs on each of <NumberOfNodes> mechines. 

```sh
./scripts/run_multi_mechines.sh <NumberOfEpochs> <DatasetName> <NumberOfGpus> <NumberOfNodes>
```