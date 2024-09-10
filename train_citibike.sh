#!/bin/bash
cuda=1
data=citibike
python train.py --dataname $data --clusters 100 --cuda $cuda --title ${data}_100
