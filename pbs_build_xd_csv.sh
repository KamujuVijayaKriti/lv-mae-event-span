#!/bin/bash
#PBS -N xd_csv
#PBS -l nodes=1:ppn=2
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -j oe
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-lvmae}
cd $PBS_O_WORKDIR
XD_ROOT=${XD_ROOT:-$HOME/XD_Violence}
python data_preprocess/build_xd_csv.py $XD_ROOT/data $XD_ROOT/langbind_embeddings $XD_ROOT
