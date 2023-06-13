#!/bin/bash
rm -r runs 
python go.py --gpu $1 --delta 0.5 --batch_size 64 --epochs 100 -ve 0.1 -vt 0.1 --fuse
python go.py --gpu $1 --delta 0.5 --batch_size 64 --epochs 100 -ve 1 -vt 1 --fuse
python go.py --gpu $1 --delta 0.5 --batch_size 64 --epochs 100 -ve 0.75 -vt 0.5 --fuse
python go.py --gpu $1 --delta 0.5 --batch_size 64 --epochs 100 -ve 0.5 -vt 0.75 --fuse