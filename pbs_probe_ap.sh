#!/bin/bash
#PBS -N ap_probe
#PBS -l nodes=1:ppn=4
#PBS -l ngpus=1
#PBS -l walltime=12:00:00
#PBS -q workq
#PBS -j oe
# Usage: qsub -N ap_<name> -v M=event_span pbs_probe_ap.sh
cd $PBS_O_WORKDIR
export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-lvmae}

RUNDIR=$(ls -d logs/prod_combined_*_e151_*ms_True_${M} | head -1)
export LVMAE_CKPT=$RUNDIR/checkpoint-150.pth
export LVMAE_AP_NAME=best_ap_${M}_seed${SEED:-0}.pth
export LVMAE_SEED=${SEED:-0}
echo "strategy=$M ckpt=$LVMAE_CKPT"
python test_ap.py
