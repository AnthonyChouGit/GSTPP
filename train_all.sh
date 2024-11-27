#!/bin/bash
cuda=7
python train.py --dataname earthquakes --clusters 100 --cuda $cuda --title earthquakes_100
python train.py --dataname covid19 --clusters 100 --cuda $cuda --title covid19_100
python train.py --dataname citibike --clusters 100 --cuda $cuda --title citibike_100
