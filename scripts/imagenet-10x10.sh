#!/bin/bash
for SEED in 0
do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach ada_gauss --seed $SEED --batch-size 256 --num-workers 8 --nepochs 200  --use-224 --datasets imagenet_subset_kaggle --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --adaptation-strategy full --lamb 10 --use-test-as-val --rotation --distillation projected --criterion ce  --normalize  --distiller mlp --adapter mlp --multiplier 32 --dump --exp-name 10x10/

done
