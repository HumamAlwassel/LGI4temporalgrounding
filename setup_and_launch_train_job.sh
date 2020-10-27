#!/bin/bash --login

DATASET=charades 
METHOD_TYPE=tgn_lgi 
CONFIG_PATH=src/experiment/options/${DATASET}/${METHOD_TYPE}/LGI.yml 
OUTPUT_ROOT=/ibex/scratch/alwassha/pytorch-experiments/lgi/${DATASET}/
NUM_WORKERS=6
FEAT_DIM=512
INIT_LR=0.0004

NUM_RUNS=1
START_RUN_ID=0

for FEATURE_TYPE in \
r2plus1d-18_features_one-head_0.0001-0.0001-0.0001-0.0001-0.004_model_5 \
r2plus1d-18_features_one-head_fc-only-0.004_model_5 \
r2plus1d-18_features_two-heads-A-noA-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.004_model_6 \
r2plus1d-18_features_two-heads-A-noA-with-global-avg-features-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.002_model_6 \
r2plus1d-18_features_two-heads-A-noA-with-global-max-features-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.002_model_6 \
r2plus1d-34_features_one-head_0.0001-0.0001-0.0001-0.0001-0.002_model_5 \
r2plus1d-34_features_one-head_fc-only-0.004_model_5 \
r2plus1d-34_features_two-heads-A-noA-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.004_model_5 \
r2plus1d-34_features_two-heads-A-noA-with-global-avg-features-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.004_model_5 \
r2plus1d-34_features_two-heads-A-noA-with-global-max-features-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.002_model_5 
do
        VIDEO_FEATURE_PATH=/ibex/scratch/alwassha/e2e-video_features/charades/features_from_activitynet_models_stride_8/${FEATURE_TYPE}.h5
        for i in $( seq $START_RUN_ID $(( START_RUN_ID + NUM_RUNS - 1 )) )
        do
                RUN_ID=run_${i}
                OUTPUT_PATH=${OUTPUT_ROOT}/${FEATURE_TYPE}/${INIT_LR}/${RUN_ID}
                JOB_NAME=LGI-${DATASET}-${FEATURE_TYPE}-${INIT_LR}-${RUN_ID}

                mkdir -p $OUTPUT_PATH
                echo $JOB_NAME

                sbatch --gres=gpu:1 --job-name=${JOB_NAME} \
                --export=ALL,CONFIG_PATH=$CONFIG_PATH,METHOD_TYPE=$METHOD_TYPE,DATASET=$DATASET,OUTPUT_PATH=$OUTPUT_PATH,FEATURE_TYPE=$FEATURE_TYPE,VIDEO_FEATURE_PATH=$VIDEO_FEATURE_PATH,FEAT_DIM=$FEAT_DIM,INIT_LR=$INIT_LR,NUM_WORKERS=$NUM_WORKERS,JOB_NAME=$JOB_NAME \
                slurm_train_lgi.sh
        done
done

