#!/bin/bash

S=$1
VAL_SAMPLES=$2
STEP=$3

echo $1 $2 $3

if [ ! -z $S ] && [ ! -z $VAL_SAMPLES ] && [ ! -z $STEP ]; then

    python3 prepare_data.py -f /ddn/beamline/Fernando/upscaling/talitas/data/0001/recon.h5 -s $STEP -p $S -o $PWD/data/label_data.h5
    python3 prepare_data.py -f /ddn/beamline/Fernando/upscaling/talitas/data/0001/recon_even.h5 -s $STEP -p $S -o $PWD/data/train_data.h5 -i
    python3 prepare_data.py -f /ddn/beamline/Fernando/upscaling/talitas/data/0019/recon.h5 -s $STEP -p $S -o $PWD/data/validation_label.h5 --sample $VAL_SAMPLES
    python3 prepare_data.py -f /ddn/beamline/Fernando/upscaling/talitas/data/0019/recon_even.h5 -s $STEP -p $S -o $PWD/data/validation.h5 --sample $VAL_SAMPLES -i

else
    echo 'Missing arguments'
    echo $0 SIZE VAL_SAMPLES STEP
    exit
fi
