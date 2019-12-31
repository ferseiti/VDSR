#!/bin/bash

S=64

    for i in {0004..0007}; do
        python3 fake_prepare_data.py --path /ddn/beamline/Fernando/upscaling/talitas/data/${i}/ -s $S -p $S -o /ddn/beamline/Fernando/upscaling/talitas/data/fake/validation/${i}_validation.h5 
    done &

    for i in {0008..0011}; do
        python3 fake_prepare_data.py --path /ddn/beamline/Fernando/upscaling/talitas/data/${i}/ -s $S -p $S -o /ddn/beamline/Fernando/upscaling/talitas/data/fake/train/${i}.h5
    done &

    for i in {0012..0015}; do
        python3 fake_prepare_data.py --path /ddn/beamline/Fernando/upscaling/talitas/data/${i}/ -s $S -p $S -o /ddn/beamline/Fernando/upscaling/talitas/data/fake/train/${i}.h5
    done &

    for i in {0016..0019}; do
        python3 fake_prepare_data.py --path /ddn/beamline/Fernando/upscaling/talitas/data/${i}/ -s $S -p $S -o /ddn/beamline/Fernando/upscaling/talitas/data/fake/train/${i}.h5
    done
