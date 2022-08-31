#!/bin/bash

#PBS -l walltime=04:00:00

#PBS -l cput=04:00:00

#PBS -l mem=6000mb

#PBS -l nodes=1

#PBS -m n

INPUT_NAME=$1

cd /unix/dune/awilkinson/extrapolation/larpixsoft
source /unix/dune2/awilkinson/miniconda3/bin/activate
conda activate larnd-sim_ROOT

OUTPUT_NAME=${INPUT_NAME%.h5}_dump.root

echo "Reading $INPUT_NAME"
echo "Writing to ${OUTPUT_NAME}..."

python export_depos_packets_toroot.py -o data/detsim_dump/${OUTPUT_NAME} data/detsim/nogaps/${INPUT_NAME}
