#!/bin/bash
MAIN_MODULE=$HOME
JOB_DIR=$MAIN_MODULE/idd20kII/
cd $JOB_DIR
module load compiler/cuda/10.2/compilervars
module load lib/cudnn_cu-9.2/7.1.4/precompiled
python train.py --model unet
