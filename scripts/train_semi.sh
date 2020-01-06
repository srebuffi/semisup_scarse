#!/usr/bin/env bash

for i in {1..10}
do

   python alternative_training.py \
          --milestones_inner 7 10 \
          --milestones_outer 60 100 \
          --dataset $1 \
          --network ResNet-18\
          --lr 0.01 \
          --nb_labels_per_class $2 \
          --proportion_CE 0.5 \
          --save_dir $3 \
          --rotnet_dir $4 \
          --seed $i

done
