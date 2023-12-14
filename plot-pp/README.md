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

Results reported below show accuracy across 11 recognition datasets averaged over 1 2 4 8 16 shots, you can download corresponding vision-only models (stage1) and joint-train models (stage2) through links given in corresponding columns.

| Dataset      | 1shots | model | 2shots | model | 4shots | model | 8shots | model | 16shots | model |
|------------  |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Caltech101   | 94.34 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EpC3d68Pux5PkGBpUdBdJlcB_8U1bE4B_7iah1MW4I89Qg?e=vxrJGJ) | 94.69 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EvgO7kJupK5HiEkQAn3ZUzgBHJg4Oyih4ku0gp0m_QrjyA?e=LpHrYl) | 95.13 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EtgwloTb4MtOppbHiMwFTJEBN4dp4jD2H-FCEmSRZG810A?e=7hS8Ec) | 95.51 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EhFiGP8jX0RApGU432Oo7I8Bsb7Tx4RbDsCUH9NKSltCng?e=hfUPjS) | 96.04 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EktGNwCnbtNHos9kgYV27tQB2eVOwpDlpP2BSnSrCdpdfw?e=Qkvilk) |
| DTD          | 54.57 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EsRgj4ZUbDRMhW21q0nIUhMBKqFdpBuObo8H19ht8cUHng?e=FnyruT) | 56.72 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EvtV_bw1y7tPoObvHnI9wLoB2dyLeczmAkUJIGO9EGCa7Q?e=bPywfG) | 62.43 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EhJ4QMsjQ6lBmoiJVh08bIYBSxAo0_0DlloxFcxNpkYu6g?e=vplGyO) | 66.49 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/Eup_f1aNqVVIlZjpZ3B9nMoBj09xJQxKfbsxWVljKNQm5w?e=1UZd4q) | 71.43 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EtUyPQbtHjVKqW4rGegEVqkBBQ15wN_Ip-p31eYYsIbdLg?e=obc9RF) |
| EuroSAT      | 65.41 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/En4S1gqR94NEra5WwycVcgcBIgNclPl6HK8Y_CtuhrZFTQ?e=iFktfL) | 76.80 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EgmVkurgSI5FnGVJb0ZRt_gBAlApnF-wNvI7LvfKRf0okQ?e=cb5Efq) | 83.21 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EpjQMS808TpDqzb2p-Utc88BRcEaTklqdKjUmoHxO32SQA?e=yLO7BH) | 88.37 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EsN7jMoKXwxEkqR-3M4j4HAB9zLC0q_PxIwe6cjk40SHmg?e=GQAVHE) | 92.00 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/ErBcLogSVD5LkNKcFU5VJVAB0_Tv17guzoUtM6Rb9W1sfA?e=b5WJfh) |
| FGVCAircraft | 28.60 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/Eg7zuRGnVslBlcMoK7YqnfcBLJtJ-V1AM114QUgHPFWx0A?e=fy73fm) | 31.14 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EsidgefYrjNIophMkt1p-BABX8043Lsdt6MfyZdJoEh2Hg?e=ECSzXF) | 35.29 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EsR_-hWn-XxIucv4DToMdGYBvs1mGqTh6oenn1fbrlwYZA?e=Bc7go3) | 41.42 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/Etw5ybVaHFtCqB7t5vFJ_-gBhkpNLu_bEsdTakyvkot1Dw?e=sOMbNO) | 46.74 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EkUgAeN7o9tOjRoCVOhGD_UBOCRAGVpEOLP6QCA7XlPLYg?e=Op1e5C) |
| Flowers102   | 80.48 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EutE09bBQNpIvWIMg5hrpI0B09YfHEQSo82KHq4y-abmpA?e=1LMdTO) | 89.81 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/Ei5owC1EViBJn2EL7lDc-dkBSLeVE-2x4KqdT_0iJpQNLA?e=SFS67Y) | 92.93 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EtUTWRQxExpOmdP74LfzTQ4BF72pNbIctQPa1D-ebiJjUw?e=Na55Tp) | 95.44 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EtJYwb-Lv0xHq5lKexyFmWEBE9TbsnfluZ3FpgdtZXkb6Q?e=Z03a0Z) | 97.56 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EmhlW-VfdxxJrNUCrTgK2BQBBu3XuyfEKqfpg7YMgEiYsw?e=XGTfWQ) |
| FOOD101      | 86.16 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/Ev2t-mQS0D5JrJj3V0EV64MB0xIwhuq04jyqYcnZE4gnvA?e=BT6dQ7) | 86.33 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EgL0VLY1ZJ9Gp1YRqofypZEB9S2tFM14wrqRgJdG6dAKkA?e=9pK0LO) | 86.46 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/Eu7tffhul4lAjI_rKQ5PHpcB6CO1MZ1KWI-pJHdzFYrXMQ?e=O51pea) | 86.58 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EqAeComPOy1JnkgrmtXa3IUBJjWFv-7SctPyCAPbxO95xQ?e=h91AqS) | 87.11 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/Eti4lZxRqeJOnvVhFZyFaLsBVJARYBUMqkUbU0U7UibwWA?e=udoWIq) |
| ImageNet     | 66.45 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EthJgxAV9adKoRLUJJFQ7kwBK1nQsl2o6y4aUWlnqcZDdQ?e=rqD9Sm) | 68.28 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EpodJ-4zuGRJg-H5L_F6zhkBtQ_rKThDhJrGQOXEDpeh3g?e=319EUh) | 70.40 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EuqHtUZPavJLoSaatKpfdl4BIEG2LS0nqRXXQ1u_ub_qlQ?e=jcAzqe) | 71.31 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EgGihjxSk1tOvYlxiIU5zpMBDpHrnrdjd0It-BX6hou48A?e=buJpC0) | 72.60 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EscvB4NMippEnLGGfFUDE5oBOwjEz39nvptGEGQGO7oiZg?e=6xRg0c) |
| OxfordPets   | 91.89 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EptQFej4VBxEgAoknhK9UBABl4d4Hcwv1vuy_VWVJkzzIg?e=8OPQSy) | 92.29 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/Ek1jb-FipqVPjMvvwugWg-8BEJ9qXYrGaMhZq-T_qg546Q?e=ctwL3b) | 92.55 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EoZX9scUDslFmCbSw6KHRHsBYc3kqot1NV72oO3h3vOCWA?e=GMNJQM) | 93.02 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/Ek7V8NufkO9JvrEMoYL91H4BBogN7ZAxMFc6xF-r_pcyig?e=91mRcd) | 93.59 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EmwyuN-CBw5OjVIPI7m4Q5kBDL8bwq57GuBjlzG3B8zeGw?e=gaPlpK) |
| StanfordCars | 68.81 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EpZsc4Arwx9Ph6Wb-likzBoBw8fxfPdxMz8mOfeB19YKPQ?e=xxARp3) | 73.17 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/Eq7is_0bJ7FEvu_06bMH-iEB3d3YcybJkj9XOB8sVpYBAg?e=P92HX6) | 76.25 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EoHWJH_0xeFIgXJywB1uFnUBIctkj90Evej0_kKzeadzJQ?e=GqmjDU) | 81.26 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EkrpCOAFhUBHuexlYhU4Ae8BbNW97rPMW60Uz8p8oOb8ag?e=2wk9US) | 84.55 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/ErRlaVwjyaVDke1k_s8iWBYBe5usZn8cr6DGtmmph2aBUA?e=Nqcubi) |
| SUN397       | 66.77 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EuOvgVWtSz5HiJdNd8XfS0kB9O2uSN5gc9DyzdERxDAf_A?e=I7zQz3) | 68.06 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/Ent0vQXNAfpLpf4lIcmY4EEBz3cuHPhENfatmeXCwNSGCA?e=pBuvyT) | 71.73 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/Eg2jxf0pEmxKm2Lw3MF-LZMBGteNHLAmOQTUEqmN7FvcXA?e=ivwlHK) | 73.93 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EpAFtEzEujhLpxguA7_q-JoBu_YwjN36Emhb8d57zc_thA?e=PYu8Z7) | 76.03 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EtXeFhvuO_tJmZaJlo44hJ8Bo0g1frAQwd2HWvzTr-MxnQ?e=94Bhwc) |
| UCF101       | 74.31 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EvJqUHG0BYpNpT28fnY8yI8BvLwCTEgU_cPqLpDatGmYKQ?e=tS5d5O) | 76.76 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EpSke3fI--pAjpjNiBPuMycBXYArQkF-ZpZdT-hKXInoDA?e=nWRFOb) | 79.76 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EndIfUyFVkNMlU0NxqRbgjoB_kCAdWjDyy4qdkhKZ7Yntg?e=Fw3Tz2) | 82.80 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/Ejv0gZpnLfRHsu9mzoMVTZ4B-tmfqBtQ9R6fUi1K1N7TlQ?e=zwQOvF) | 85.34 | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zhengqing_gao_mbzuai_ac_ae/EiGTvdSwgZVOhm9HeXvLxQcBrjpwdb5zvaoKqG7qkzAT_A?e=SJ8tUR) |
 
You can use `/scripts/evaluation.sh` to evaluate `PLOT++` by loading our released models from links shown above, whose structure is

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

To evaluate the first-stage models, you should firstly gain the stage1 model, and store it in `your_pretrain_path`, change `your_data_path`, `your_pretrain_path` and `your_work_path` in `scripts/evaluation.sh`, then you can run the command `bash evaluation.sh DATASET 4` under `plot-pp/scripts/`. For example `bash evaluation.sh imagenet 4`. To evaluate the stage2 models, you should firstly gain the second-stage model and repeat what you do when evaluating the first-stage model.
