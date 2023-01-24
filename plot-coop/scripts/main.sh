#!/bin/bash

cd ..

# custom config
DATA=your_data_path
TRAINER=CoOp 

DATASET=$1
CFG=rn50  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
# SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
N=$2  # number of proxy

for SHOTS in 1 2 4 8 16
do
for SEED in 1 2 3
do
DIR=your_work_path/plot-coop/output/OP_N${N}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
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
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    MODEL.N ${N} 
fi
done
done