#! /bin/bash

options=$1		# path to configuration file without ".yml"
m_type=$2		# method type
dataset=$3		# dataset type
gpu_id=$4		# ID of GPU 
num_thread=$5	# number of threads for data loader, use 4
debug=$6		# flag for debug mode, 1 if debug else 0
output_path=$7		# output_path
feature_type=$8		# feature_type
video_feature_path=$9		# video_feature_path

if [ ${debug} -eq 1 ]
then
	#CUDA_LAUNCH_BLOCKING=1 python -m src.experiment.train \
	CUDA_VISIBLE_DEVICES=${gpu_id} python -m src.experiment.train \
		--config_path src/experiment/options/${dataset}/${m_type}/${options}.yml \
		--method_type ${m_type} \
		--dataset ${dataset} \
		--output_path ${output_path}\
		--feature_type ${feature_type}\
		--video_feature_path ${video_feature_path}\
		--num_workers ${num_thread} \
		--debug_mode 
else
	CUDA_VISIBLE_DEVICES=${gpu_id} python -m src.experiment.train \
		--config_path src/experiment/options/${dataset}/${m_type}/${options}.yml \
		--method_type ${m_type} \
		--dataset ${dataset} \
		--output_path ${output_path}\
		--feature_type ${feature_type}\
		--video_feature_path ${video_feature_path}\
		--num_workers ${num_thread} 
fi
