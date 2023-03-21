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
Results reported below show accuracy across 11 recognition datasets averaged over 1 2 4 8 16 shots, you can download corresponding vision-only models (stage1) and joint-train models (stage2) through links given in last 2 columns.

| Dataset      | 1shots | links | 2shots | links | 4shots | links | 8shots | links | 16shots | links |
|------------  |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Caltech101   | 93.14 | [stage1]();[stage2]() | 94.69 | [stage1]();[stage2]() | 95.13 | [stage1]();[stage2]() | 95.51 | [stage1]();[stage2]() | 96.04 | [stage1]();[stage2]() |
| DTD          | 54.57 | [stage1]();[stage2]() | 56.72 | [stage1]();[stage2]() | 62.43 | [stage1]();[stage2]() | 66.49 | [stage1]();[stage2]() | 71.43 | [stage1]();[stage2]() |
| EuroSAT      | 65.41 | [stage1]();[stage2]() | 76.80 | [stage1]();[stage2]() | 83.21 | [stage1]();[stage2]() | 88.37 | [stage1]();[stage2]() | 92.00 | [stage1]();[stage2]() |
| FGVCAircraft | 28.60 | [stage1]();[stage2]() | 31.14 | [stage1]();[stage2]() | 35.29 | [stage1]();[stage2]() | 41.42 | [stage1]();[stage2]() | 46.74 | [stage1]();[stage2]() |
| Flowers102   | 80.48 | [stage1]();[stage2]() | 89.81 | [stage1]();[stage2]() | 92.93 | [stage1]();[stage2]() | 95.44 | [stage1]();[stage2]() | 97.56 | [stage1]();[stage2]() |
| FOOD101      | 86.16 | [stage1]();[stage2]() | 86.33 | [stage1]();[stage2]() | 86.46 | [stage1]();[stage2]() | 86.58 | [stage1]();[stage2]() | 87.11 | [stage1]();[stage2]() |
| ImageNet     | 66.45 | [stage1]();[stage2]() | 68.28 | [stage1]();[stage2]() | 70.40 | [stage1]();[stage2]() | 71.31 | [stage1]();[stage2]() | 72.60 | [stage1]();[stage2]() |
| OxfordPets   | 91.89 | [stage1]();[stage2]() | 92.29 | [stage1]();[stage2]() | 92.55 | [stage1]();[stage2]() | 93.02 | [stage1]();[stage2]() | 93.59 | [stage1]();[stage2]() |
| StanfordCars | 68.81 | [stage1]();[stage2]() | 73.17 | [stage1]();[stage2]() | 76.25 | [stage1]();[stage2]() | 81.26 | [stage1]();[stage2]() | 84.55 | [stage1]();[stage2]() |
| SUN397       | 66.77 | [stage1]();[stage2]() | 68.06 | [stage1]();[stage2]() | 71.73 | [stage1]();[stage2]() | 73.93 | [stage1]();[stage2]() | 76.03 | [stage1]();[stage2]() |
| UCF101       | 74.31 | [stage1]();[stage2]() | 76.76 | [stage1]();[stage2]() | 79.76 | [stage1]();[stage2]() | 82.80 | [stage1]();[stage2]() | 85.34 | [stage1]();[stage2]() |

## Evaluation


We provided first stage models and second stage models for each dataset with 16 shots. You can use `/scripts/evaluation.sh` to evaluate `PLOT++` by loading our [released models](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/Evwfz9JwqxZDuSEuSC6QSEwBMx_Old9XJ4zziZEXpwQnIw?e=s55Ezl), whose structure is

```
plot++_models
|–– caltech101/
|   |–– 1shots/
|   |   |–– stage1/
|   |–– |–– |–– seed1/
|   |   |–– stage2/
|   |   |–– |–– seed1/
|   |   |–– |–– seed2/
|   |   |–– |–– seed3/
```

To evaluate the first-stage models, you should firstly gain the first-stage model, and store it in `your_pretrain_path`, change `your_data_path`, `your_pretrain_path` and `your_work_path` in `scripts/evaluation.sh`, then you can run the command `bash evaluation.sh DATASET 4` under `plot-pp/scripts/`. For example `bash evaluation.sh imagenet 4`. To evaluate the second-stage models, you should firstly gain the second-stage model and repeat what you do when evaluating the first-stage model.
