# PLOT-CoOp
This folder contains the implementation of the PLOT method on prompt learning.

This code is built on top of [CoOp](https://github.com/KaiyangZhou/CoOp).

## Build Environment
Following CoOp, it is needed to install the `dassl` environment. You can follow the scripts [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. Then, you can install other package by running `pip install -r requirements.txt` (this should be done when `dassl` is activated).


## Install Dataset
Please follow the instructions [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to construct the datasets.


## Run Scripts


The running scripts are in `scripts/`. `cd ./scripts` and change the `your_data_path` and `your_work_path` in `scripts/main.sh`
Then, you can run the commands `bash main.sh DATASET N` under `CoOp/scripts/`.

`DATASET` takes as input a dataset name, like `caltech101`. 

`N` is the number of prompts, such as `4`.

### Results

Same as CoOp, you can find the results from `output/` whose structure is

```
output
|–– OP_N4/caltech101/
|   |–– PLOT/
|   |   |–– rn50_16shots/
|   |   |   |–– nctx16_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
|   |   |–– rn50_8shots/
|   |   |   |–– nctx16_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
```

### Visualization

If you would like to visualize the transport plan with the attention map. 
Please use the following scripts to visualize one transport plan `T` ($T \in \mathcal{R}^{7\times7}$):
```
import cv2
viz_atten = cv2.applyColorMap(T, cv2.COLORMAP_JET)  
viz_atten_224 = cv2.resize(viz_atten, (224, 224), interpolation=cv2.INTER_CUBIC)
```
