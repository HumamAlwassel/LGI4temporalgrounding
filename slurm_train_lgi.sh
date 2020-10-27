#!/bin/bash --login
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%A-%3a.out
#SBATCH --error=logs/%x-%A-%3a.err
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=40G
#SBATCH --mail-type=FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user=humam.alwassel@kaust.edu.sa
#SBATCH -A conf-gpu-2020.11.23

set -ex

hostname
nvidia-smi
env

conda activate tg

python -m src.experiment.train \
--config_path ${CONFIG_PATH} \
--method_type ${METHOD_TYPE} \
--dataset ${DATASET} \
--output_path ${OUTPUT_PATH} \
--feature_type ${FEATURE_TYPE} \
--video_feature_path ${VIDEO_FEATURE_PATH} \
--feat_dim ${FEAT_DIM} \
--init_lr ${INIT_LR} \
--num_workers ${NUM_WORKERS}
