#!/bin/bash
for SEED in 1
do
  for VAL in 10
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach ada_gauss --seed $SEED --batch-size 256 --num-workers 4 --nepochs 200 --datasets tiny --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --adaptation-strategy full --S 64 --lamb 10 --use-test-as-val --criterion ce --distillation projected  --rotation --normalize --multiplier 32  --distiller mlp --adapter mlp --exp-name 10x10/
  done
done
