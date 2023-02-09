# PLOT-Adapter
This folder contains the implementation of the PLOT method on adpaters.

This code is built on top of [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter).

## Build Environment
Following [PLOT-CoOp](https://github.com/CHENGY12/PLOT/tree/main/plot-coop) to build the evironment of CoOp. Then, you can install other packages of PLOT-Adapter by running `pip install -r requirements.txt` in current folder.


## Install Dataset
Please follow the instructions [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to construct the datasets.


## Build Caches
Change the data root path in `configs/dataset.yaml` to your own path
Please feel free to download the cached models [here](https://drive.google.com/file/d/1B3BNfFKwFciDwR9uwohAh0IxmxWAo5B9/view?usp=sharing). Extract the pretrained models of each datasets and put them into caches folder, whose structure is

```
plot-adapter
|–– caches
|   |–– caltech101/
|   |   |–– shot1
|   |   |–– shot2
|   |   |–– shot4
|   |   |–– shot8
|   |   |–– shot16
|   |–– dtd/
|   |   |–– shot1
|   |   |–– shot2
|   |   |–– shot4
|   |   |–– shot8
|   |   |–– shot16
```

## Run Scripts

For ImageNet dataset:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_imagenet.py --config configs/imagenet.yaml
```
For other 10 datasets:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/dataset.yaml
```


