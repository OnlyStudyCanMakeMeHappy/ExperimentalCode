#!/bin/bash
python main.py --gpu 0 --data FGNET --k 5 --logdir logs/FGNET
python main.py --gpu 0 --data Adience --fuse --aug --record
