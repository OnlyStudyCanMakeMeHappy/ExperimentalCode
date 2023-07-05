#!/bin/bash
python main.py --gpu $1 --data Adience  --delta 0.2 -vt 0.2 -ve 0.2 -ls multi_step --lr 1e-4 -j 8 --val_epoch 1 --record
python main.py --gpu $1 --data Adience  --delta 0.2 -vt 0.2 -ve 0.2 -ls multi_step --lr 1e-4 --fuse -j 8 --val_epoch 1
python main.py --gpu $1 --data Adience  --delta 0.2 -vt 0.2 -ve 0.2 -ls cosine_anneal --lr 1e-4 --fuse -j 8 --val_epoch 1
python main.py --gpu 1 --data Adience  -ls multi_step 20 40 50 --lr 1e-4 -j 8 --val_epoch 1 --record