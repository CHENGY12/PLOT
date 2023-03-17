#!/bin/bash

cd ..

# custom config
DATA=your_data_path
PRETRAIN_DIR=your_pretrain_path
TRAINER=PLOTPP
DATASET=$1
CFG=vit  # config file
CTP=end  # class token position (end or middle)
M=$2  # number of vision prompts
N=4  # number of text prompts
NCTX=4  # number of context tokens
NCTX_V=2  # number of vision context tokens
# SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

for SHOTS in 1 2 4 8 16
do
for SEED in 1 2 3
do
DIR=your_work_path/plot-pp/output_joint/OP_N${N}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --device cuda:1 \
    TRAINER.PLOTPP.N_CTX ${NCTX} \
    TRAINER.PLOTPP.CSC ${CSC} \
    TRAINER.PLOTPP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.PLOTPP.M ${M}\
    TRAINER.PLOTPP.N ${N} \
    TRAINER.PLOTPP.N_CTX_V ${NCTX_V} \
    TRAINER.PLOTPP.CTX_INIT True\
    TRAINER.PLOTPP.TRADE_OFF "False"\
    TRAINER.PLOTPP.PRETRAIN_DIR ${PRETRAIN_DIR}\
    TRAINER.PLOTPP.MODEL_UPD "joint"
fi
done
done
