# PLOT++
This folder contains the implementation of the PLOT++ method which extends PLOT to support the ViT backbones. Models are coming (in two days). 


## Build Environment
Following CoOp, it is needed to install the 'dassl' environment. You can follow the scripts [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. Then, you can install other package by running `pip install -r requirements.txt` (this should be done when `dassl` is activated).


## Install Dataset
Please follow the instructions [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to construct the datasets.


## Run Scripts


Our `PLOT++` includes two stages, both two running scripts are in `scripts/`. 

To run the first stage, you should firstly run `cd ./scripts` and change `your_data_path` and `your_work_path` in `scripts/main_visiononly.sh`, then you can run the commands `bash main_visiononly.sh DATASET M` under `plot-pp/scripts/`. For example `bash bash main_visiononly.sh caltech101 4`

`DATASET` takes as input a dataset name, like `caltech101`. 

M is the number of vision prompts, such as 4.

After the first stage, you can find the results from `output_visiononly/`, whose structure is
```
output
|–– OP_N4/caltech101/
|   |–– PLOTPP/
|   |   |–– vit_16shots/
|   |   |   |–– nctx4_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
|   |   |–– vit_8shots/
|   |   |   |–– nctx4_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
```
you can find the first-stage pretrained model from `seed1/visiononly/model.pth.tar-50`.

To run the second stage, you should change `your_data_path`, `your_pretrain_path` and `your_work_path` in `scripts/main_joint.sh`, then you can run the commands `bash main_joint.sh DATASET M` under `plot-pp/scripts/`. For example `bash main_joint.sh caltech101 4`

Note that `DATASET` and `M` should have the same values as the fist stage.

### Results

You can find the results from `output_joint/` whose structure is

```
output_joint
|–– OP_N4/caltech101/
|   |–– PLOTPP/
|   |   |–– vit_16shots/
|   |   |   |–– nctx4_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
|   |   |–– vit_8shots/
|   |   |   |–– nctx4_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
```

## Evaluation


You can use `/scripts/evaluation.sh` to reproduce `PLOT++` by loading our realeased model. You should firstly change `your_data_path`, `your_pretrain_path` and `your_work_path` in `scripts/evaluation.sh`, then you can run the command `bash evaluation.sh DATASET M` under `plot-pp/scripts/`. For example `bash evaluation.sh imagenet 4`.
