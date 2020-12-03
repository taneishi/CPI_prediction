#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -N cpi
#PBS -j oe
#PBS -o output.log

cd ${PBS_O_WORKDIR}
mkdir -p model

python main.py --epochs 10
