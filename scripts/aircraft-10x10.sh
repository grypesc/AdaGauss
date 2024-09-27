#!/bin/bash
for SEED in 0
do
  for VAL in 10
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach ada_gauss --seed $SEED --batch-size 256 --num-workers 8 --nepochs 200 --datasets aircraft --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --adaptation-strategy full --S 32 --lamb 0.3  --use-test-as-val --criterion ce --normalize --multiplier 32 --distiller mlp --adapter mlp --pretrained-net --use-224 --exp-name 10x10/v1
  done
done
