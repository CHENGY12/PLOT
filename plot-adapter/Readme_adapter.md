This code is built on top of CoOp. Thus, it is needed to install the `dassl` environment which is used for CoOp. You can follow the scripts [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. Then, you can install other package by running `pip install -r requirements.txt` (this should be done when `dassl` is activated).


### Install DATASET
- Create a folder named `caltech-101/` under `$DATA`.
- Download `101_ObjectCategories.tar.gz` from http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz and extract the file under `$DATA/caltech-101`.
- Download `split_zhou_Caltech101.json` from this [link](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and put it under `$DATA/caltech-101`. 

The directory structure should look like
```
caltech-101/
|–– 101_ObjectCategories/
|–– split_zhou_Caltech101.json
```

### Run Scripts

The running scripts in `scripts/`. You can run the commands `bash main.sh caltech101 rn50 end 16 False 4` under `CoOp/scripts/`.

`DATASET` takes as input a dataset name, like `caltech101`. 

`CFG` means which config file to use, such as `rn50`.

`CTP` is class token position. We fix it as `end`.

`NCTX` is number of context tokens. We fix it as `16`.

`CSC` denotes whether to use class-specific context. We use `False`.

`N` is the number of prompts, such as `4`.
