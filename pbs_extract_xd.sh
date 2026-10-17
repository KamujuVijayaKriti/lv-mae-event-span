#!/bin/bash
#PBS -N xd_lb
#PBS -l nodes=1:ppn=4
#PBS -l ngpus=1
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -j oe
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-lvmae}
cd $PBS_O_WORKDIR
export PYTHONPATH=$PBS_O_WORKDIR:$PYTHONPATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
XD_ROOT=${XD_ROOT:-$HOME/XD_Violence}
python -X faulthandler data_preprocess/extract_xd.py $XD_ROOT/data/video $XD_ROOT/langbind_embeddings ${TASK:-0} ${NTASK:-1}
