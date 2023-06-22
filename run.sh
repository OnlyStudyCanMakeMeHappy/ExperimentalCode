#!/bin/bash
python go.py --gpu $1 --delta 0.75 --batch_size 20 --epochs 100
python go.py --gpu $1 --delta 0.75 --batch_size 32 --epochs 100
python go.py --gpu $1 --delta 0.75 --batch_size 64 --epochs 100
python go.py --gpu $1 --delta 0.5 --batch_size 64 --epochs 100
python go.py --gpu $1 --delta 0.05 --batch_size 64 --epochs 100
python go.py --gpu $1 --delta 0.1 --batch_size 64 --epochs 100
python go.py --gpu $1 --delta 1.0 --batch_size 64 --epochs 100
python go.py --gpu $1 --delta 0.75 -vt 0.75 -ve 0.75 --batch_size 64 --epochs 100 --fuse
python go.py --gpu $1 --delta 0.75 -vt 1.0 -ve 1.0 --batch_size 64 --epochs 100 --fuse
