#!/bin/bash
#PBS -N lvmae_pre
#PBS -l nodes=1:ppn=4
#PBS -l ngpus=1
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -j oe
# Usage:  qsub -v MASKING=event_span,EPOCHS=151 pbs_pretrain_masking.sh
# MASKING: random | paper | repo | stochastic | event_span

cd $PBS_O_WORKDIR
export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-lvmae}

export LVMAE_MASKING=event_span
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
EPOCHS=${EPOCHS:-151}
WARMUP=${WARMUP:-40}
echo "masking=$LVMAE_MASKING epochs=$EPOCHS host=$(hostname)"
nvidia-smi

python main_pretrain.py --masking_strategy --mask_ratio 0.5 \
    --batch_size ${BATCH:-8} --accum_iter ${ACCUM:-2} --epochs $EPOCHS --warmup_epochs $WARMUP \
    --input_size 1024 --depth 32 --num_heads 16 \
    --decoder_embed_dim 512 --decoder_depth 8 --decoder_num_heads 16 \
    --num_patches 256 --world_size 1 --num_workers 4 --seed ${PSEED:-0}
