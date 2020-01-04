#!/bin/bash

S=64

    for i in {0000..0003}; do
        python3 prepare_data.py --path /ddn/beamline/Fernando/upscaling/talitas/data/${i}/ -s $S -p $S -o /ddn/beamline/Fernando/upscaling/talitas/data/'test'/${i}.h5 -b 21
    done &

    for i in {0004..0007}; do
        python3 prepare_data.py --path /ddn/beamline/Fernando/upscaling/talitas/data/${i}/ -s $S -p $S -o /ddn/beamline/Fernando/upscaling/talitas/data/validation/${i}_validation.h5 -b 21
    done &

    for i in {0008..0011}; do
        python3 prepare_data.py --path /ddn/beamline/Fernando/upscaling/talitas/data/${i}/ -s $S -p $S -o /ddn/beamline/Fernando/upscaling/talitas/data/train/${i}.h -b 215
    done &

    for i in {0012..0015}; do
        python3 prepare_data.py --path /ddn/beamline/Fernando/upscaling/talitas/data/${i}/ -s $S -p $S -o /ddn/beamline/Fernando/upscaling/talitas/data/train/${i}.h -b 215
    done &

    for i in {0016..0019}; do
        python3 prepare_data.py --path /ddn/beamline/Fernando/upscaling/talitas/data/${i}/ -s $S -p $S -o /ddn/beamline/Fernando/upscaling/talitas/data/train/${i}.h -b 215
    done
