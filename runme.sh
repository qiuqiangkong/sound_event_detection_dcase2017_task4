#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR="/vol/vssp/msos/qk/datasets/dcase2017/task4/dataset_root"

# You need to modify this path to your workspace to store features and models
WORKSPACE="/vol/vssp/msos/qk/workspaces/transfer_to_other_datasets/transfer_to_dcase2017_task4"

# Pack waveforms to hdf5
python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='testing'

python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='evaluation'

python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='training'

# Set model type to one of ['Cnn_9layers_FrameMax', 'Cnn_9layers_FrameAvg', 
# 'Cnn_9layers_FrameAtt', 'Cnn_9layers_Gru_FrameAvg', 'Cnn_9layers_Gru_FrameAtt', 
# 'Cnn_9layers_Transformer_FrameAvg', Cnn_9layers_Transformer_FrameAtt]
MODEL_TYPE="Cnn_9layers_FrameAvg"
FOLD=1  # Always set FOLD to 1

# Train
CUDA_VISIBLE_DEVICES=0 python3 pytorch/pytorch_main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=$FOLD --model_type=$MODEL_TYPE --pretrained_checkpoint_path= --loss_type=clip_bce --augmentation='mixup' --learning_rate=1e-3 --batch_size=32 --few_shots=-1 --random_seed=1000 --resume_iteration=0 --stop_iteration=50000 --cuda

CUDA_VISIBLE_DEVICES=0 python3 pytorch/pytorch_main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=$FOLD --model_type=$MODEL_TYPE --pretrained_checkpoint_path= --loss_type=clip_bce --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --few_shots=-1 --random_seed=1000 --resume_iteration=50000 --stop_iteration=60000 --cuda

# Optimize thresholds for AT and SED
python3 utils/auto_thresholds.py optimize_thresholds --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=$FOLD --model_type=$MODEL_TYPE --loss_type=clip_bce --augmentation='mixup' --learning_rate=1e-3 --batch_size=32 --few_shots=-1 --random_seed=1000 --iteration=60000

# Calculate metrics using the optimal thresholds
python3 utils/auto_thresholds.py calculate_metrics --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=$FOLD --model_type=$MODEL_TYPE --loss_type=clip_bce --augmentation='mixup' --learning_rate=1e-3 --batch_size=32 --few_shots=-1 --random_seed=1000 --iteration=60000
