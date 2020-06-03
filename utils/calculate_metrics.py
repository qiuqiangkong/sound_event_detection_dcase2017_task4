import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import time
import sklearn
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt

from utilities import (get_filename, create_folder, 
    frame_prediction_to_event_prediction, write_submission, official_evaluate)
import config


def calculate_precision_recall_f1(y_true, output, thresholds, average='micro'):
    """Calculate precision, recall, F1."""
    if y_true.ndim == 3:
        (N, T, F) = y_true.shape
        y_true = y_true.reshape((N * T, F))
        output = output.reshape((N * T, F))

    classes_num = y_true.shape[-1]
    binarized_output = np.zeros_like(output)

    for k in range(classes_num):
        binarized_output[:, k] = (np.sign(output[:, k] - thresholds[k]) + 1) // 2

    if average == 'micro':
        precision = metrics.precision_score(y_true.flatten(), binarized_output.flatten())
        recall = metrics.recall_score(y_true.flatten(), binarized_output.flatten())
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1
    
    else:
        raise Exception('Incorrect argument!')


def get_metric(results, metric_type):
    if metric_type == 'f1':
        return results['overall']['f_measure']['f_measure']
    elif metric_type == 'er':
        return results['overall']['error_rate']['error_rate']
    elif metric_type == 'precision':
        return results['overall']['f_measure']['precision']
    elif metric_type == 'recall':
        return results['overall']['f_measure']['recall']


def calculate_metrics(args):
    """Calculate metrics.

    Args:
      dataset_dir: str
      workspace: str
      holdout_fold: '1'
      model_type: str, e.g., 'Cnn_9layers_Gru_FrameAtt'
      loss_type: str, e.g., 'clip_bce'
      augmentation: str, e.g., 'mixup'
      batch_size: int
      iteration: int
      data_type: 'test' | 'evaluate'
      at_thresholds: bool
      sed_thresholds: bool
    """
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    filename = args.filename
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    batch_size = args.batch_size
    iteration = args.iteration
    data_type = args.data_type
    at_thresholds = args.at_thresholds
    sed_thresholds = args.sed_thresholds

    classes_num = config.classes_num

    # Paths
    if data_type == 'test':
        reference_csv_path = os.path.join(dataset_dir, 'metadata', 
            'groundtruth_strong_label_testing_set.csv')
    
    elif data_type == 'evaluate':
        reference_csv_path = os.path.join(dataset_dir, 'metadata', 
            'groundtruth_strong_label_evaluation_set.csv')
        
    prediction_path = os.path.join(workspace, 'predictions', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '{}_iterations.prediction.{}.pkl'.format(iteration, data_type))
    
    tmp_submission_path = os.path.join(workspace, '_tmp_submission', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '_submission.csv')

    # Load thresholds
    if at_thresholds:
        at_thresholds_path = os.path.join(workspace, 'opt_thresholds', 
            '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
            'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
            '{}_iterations.at.test.pkl'.format(iteration))
        at_thresholds = pickle.load(open(at_thresholds_path, 'rb'))
    else:
        at_thresholds = [0.3] * classes_num

    if sed_thresholds:
        sed_thresholds_path = os.path.join(workspace, 'opt_thresholds', 
            '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
            'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
            '{}_iterations.sed.test.pkl'.format(iteration))
        sed_thresholds = pickle.load(open(sed_thresholds_path, 'rb'))
    else:
        sed_thresholds = {
            'audio_tagging_threshold': 0.5, 
            'sed_high_threshold': 0.3, 
            'sed_low_threshold': 0.1, 
            'n_smooth': 10, 
            'n_salt': 10}

    # Load predictions
    output_dict = pickle.load(open(prediction_path, 'rb'))

    print('------ Audio tagging results ------')
    # Macro mAP
    mAP = metrics.average_precision_score(output_dict['target'], 
        output_dict['clipwise_output'], average='macro')
    
    # Micro precision, recall, F1
    (precision, recall, f1) = calculate_precision_recall_f1(
        output_dict['target'], output_dict['clipwise_output'], 
        thresholds=at_thresholds)

    print('Macro mAP: {:.3f}'.format(mAP))
    print('Micro precision: {:.3f}'.format(precision))
    print('Micro recall: {:.3f}'.format(recall))
    print('Micro F1: {:.3f}'.format(f1))

    print('------ Sound event detection ------')

    predict_event_list = frame_prediction_to_event_prediction(output_dict, 
        sed_thresholds)

    # Write predicted events to submission file
    write_submission(predict_event_list, tmp_submission_path)

    # SED with official tool
    results = official_evaluate(reference_csv_path, tmp_submission_path)
    
    sed_precision = get_metric(results, 'precision')
    sed_recall = get_metric(results, 'recall')
    sed_f1 = get_metric(results, 'f1')
    sed_er = get_metric(results, 'er')

    print('Micro precision: {:.3f}'.format(sed_precision))
    print('Micro recall: {:.3f}'.format(sed_recall))
    print('Micro F1: {:.3f}'.format(sed_f1))
    print('Micro ER: {:.3f}'.format(sed_er))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_calculate_metrics = subparsers.add_parser('calculate_metrics')
    parser_calculate_metrics.add_argument('--dataset_dir', type=str, required=True)
    parser_calculate_metrics.add_argument('--workspace', type=str, required=True)
    parser_calculate_metrics.add_argument('--filename', type=str, required=True)
    parser_calculate_metrics.add_argument('--holdout_fold', type=str, choices=['1', 'none'], required=True)
    parser_calculate_metrics.add_argument('--model_type', type=str, required=True)
    parser_calculate_metrics.add_argument('--loss_type', type=str, required=True)
    parser_calculate_metrics.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_calculate_metrics.add_argument('--batch_size', type=int, required=True)
    parser_calculate_metrics.add_argument('--iteration', type=int, required=True)
    parser_calculate_metrics.add_argument('--data_type', type=str, choices=['test', 'evaluate'], required=True)
    parser_calculate_metrics.add_argument('--at_thresholds', action='store_true', default=False)
    parser_calculate_metrics.add_argument('--sed_thresholds', action='store_true', default=False)

    args = parser.parse_args()

    if args.mode == 'calculate_metrics':
        calculate_metrics(args)

    else:
        raise Exception('Error argument!')
