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
MODEL_TYPE="Cnn_9layers_Gru_FrameAtt"

# ------ Train ------
python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --learning_rate=1e-3 --batch_size=32 --resume_iteration=0 --stop_iteration=50000 --cuda

python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=50000 --stop_iteration=70000 --cuda

# ------ Inference and dump predicted probabilites ------
python3 pytorch/main.py inference_prob --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --iteration=70000 --cuda

# ------ Optimize thresholds ------
# Optimize audio tagging thresholds
python3 utils/optimize_thresholds.py optimize_at_thresholds  --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --filename='main' --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --iteration=70000

# Optimize sound event detection thresholds
python3 utils/optimize_thresholds.py optimize_sed_thresholds  --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --filename='main' --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --iteration=70000

# ------ Calculate metrics ------
# Calculate statistics without automatic threshold optimization
python3 utils/calculate_metrics.py calculate_metrics --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --filename='main' --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --iteration=70000 --data_type='evaluate'

# Calculate statistics without automatic threshold optimization
python3 utils/calculate_metrics.py calculate_metrics --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --filename='main' --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --iteration=70000 --data_type='evaluate' --at_thresholds --sed_thresholds

# Plot statistics for paper
python3 utils/plot_for_paper.py plot_clipwise_at_sed --workspace=$WORKSPACE --data_type=evaluate
python3 utils/plot_for_paper.py plot_best_model_17_classes --workspace=$WORKSPACE --data_type=evaluate
python3 utils/plot_for_paper.py plot_precision_recall_curve --workspace=$WORKSPACE