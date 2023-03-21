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

| Dataset      | 1shots | 2shots | 4shots | 8shots | 16shots |
|------------  |:-----:|:-----:|:-----:|:-----:|:-----:|
| Caltech101   | 93.14 | 94.69 | 95.13 | 95.51 | 96.04 |
| DTD          | 54.57 | 56.72 | 62.43 | 66.49 | 71.43 |
| EuroSAT      | 65.41 | 76.80 | 83.21 | 88.37 | 92.00 |
| FGVCAircraft | 28.60 | 31.14 | 35.29 | 41.42 | 46.74 |
| Flowers102   | 80.48 | 89.81 | 92.93 | 95.44 | 97.56 |
| FOOD101      | 86.16 | 86.33 | 86.46 | 86.58 | 87.11 |
| ImageNet     | 66.45 | 68.28 | 70.40 | 71.31 | 72.60 |
| OxfordPets   | 91.89 | 92.29 | 92.55 | 93.02 | 93.59 |
| StanfordCars | 68.81 | 73.17 | 76.25 | 81.26 | 84.55 |
| SUN397       | 66.77 | 68.06 | 71.73 | 73.93 | 76.03 |
| UCF101       | 74.31 | 76.76 | 79.76 | 82.80 | 85.34 |

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
