#!/bin/bash
for SEED in 1
do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach ada_gauss --seed 1 --batch-size 128 --num-workers 8 --nepochs 200 --datasets cub200 --num-tasks 20 --nc-first-task 10 --lr 0.1 --lr-adapter 0.1 --weight-decay 5e-4 --adaptation-strategy full --S 32  --lamb 1  --use-test-as-val --criterion ce --normalize --multiplier 32 --distiller mlp --adapter mlp --pretrained-net --use-224 --exp-name 20x10/v5
done