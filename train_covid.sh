#!/bin/bash
cuda=6
data=covid19
# python train.py --dataname $data --clusters 10 --cuda $cuda --title ${data}_10
# python train.py --dataname $data --clusters 30 --cuda $cuda --title ${data}_30
python train.py --dataname $data --clusters 50 --cuda $cuda --title ${data}_50
python train.py --dataname $data --clusters 70 --cuda $cuda --title ${data}_70
python train.py --dataname $data --clusters 90 --cuda $cuda --title ${data}_90

