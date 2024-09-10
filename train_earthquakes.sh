#!/bin/bash
cuda=7
data=earthquakes
python train.py --dataname $data --clusters 100 --cuda $cuda --title ${data}_100
