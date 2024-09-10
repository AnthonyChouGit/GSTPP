#!/bin/bash
cuda=6
data=covid19
python train.py --dataname $data --clusters 100 --cuda $cuda --title ${data}_100
