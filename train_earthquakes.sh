#!/bin/bash
cuda=7
data=earthquakes
python train.py --dataname $data --clusters 10 --cuda $cuda --title ${data}_10_freeze20 --node_freeze 20
python train.py --dataname $data --clusters 30 --cuda $cuda --title ${data}_30_freeze20 --node_freeze 20
python train.py --dataname $data --clusters 50 --cuda $cuda --title ${data}_50_freeze20 --node_freeze 20
python train.py --dataname $data --clusters 70 --cuda $cuda --title ${data}_70_freeze20 --node_freeze 20
python train.py --dataname $data --clusters 90 --cuda $cuda --title ${data}_90_freeze20 --node_freeze 20
