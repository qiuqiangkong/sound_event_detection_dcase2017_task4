import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import sklearn
import _pickle as cPickle
from sklearn import metrics
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=1000, precision=3, suppress=True)

from opt import Adam
from utilities import (get_filename, create_folder, load_prediction, 
    frame_prediction_to_event_prediction, write_submission, official_evaluate)
import config


def calculate_f1(y_true, output, thresholds, average):
    """Calculate F1 score.

    Args:
      y_true: (N, (optional)frames_num], classes_num)
      output: (N, (optional)[frames_num], classes_num)
      thresholds: (classes_num,), initial thresholds
      average: 'micro' | 'macro'
    """
    if y_true.ndim == 3:
        (N, T, F) = y_true.shape
        y_true = y_true.reshape((N * T, F))
        output = output.reshape((N * T, F))

    classes_num = y_true.shape[-1]
    binarized_output = np.zeros_like(output)

    for k in range(classes_num):
        binarized_output[:, k] = (np.sign(output[:, k] - thresholds[k]) + 1) // 2

    if average == 'micro':
        return metrics.f1_score(y_true.flatten(), binarized_output.flatten())
    
    f1_array = []
    for k in range(classes_num):
        f1_array.append(metrics.f1_score(y_true[:, k], binarized_output[:, k]))

    if average == 'macro':
        return np.average(f1_array)
    elif average is None:
        return f1_array
    else:
        raise Exception('Incorrect argument!')


def calculate_at_gradient(y_true, output, thresholds, average):
    """Calculate gradient of thresholds numerically.

    Args:
      y_true: (N, (optional)frames_num], classes_num)
      output: (N, (optional)[frames_num], classes_num)
      thresholds: (classes_num,), initial thresholds
      average: 'micro' | 'macro'

    Returns:
      grads: vector
    """
    f1 = calculate_f1(y_true, output, thresholds, average)
    delta = 0.01
    grads = []

    for k, threshold in enumerate(thresholds):
        new_thresholds = thresholds.copy()
        delta = 0.01
        cnt = 0
        while cnt < 10:
            cnt += 1
            new_thresholds[k] += delta
            f1_new = calculate_f1(y_true, output, new_thresholds, average)

            if f1_new != f1:
                break

        grad = (f1_new - f1) / (delta * cnt)
        grads.append(grad)

    return grads


def optimize_at_with_gd(y_true, output, thresholds, average):
    """Optimize thresholds for AT.

    Args:
      y_true: (N, (optional)frames_num], classes_num)
      output: (N, (optional)[frames_num], classes_num)
      thresholds: (classes_num,), initial thresholds
      average: 'micro' | 'macro'

    Returns:
      metric: float
      thresholds: vector
    """
    opt = Adam()
    opt.alpha = 1e-2
    for i in range(100):
        grads = calculate_at_gradient(y_true, output, thresholds, average)
        grads = [-e for e in grads]
        thresholds = opt.GetNewParams(thresholds, grads)
        metric = calculate_f1(y_true, output, thresholds, average)
        print('Iteration: {}, Score: {:.3f}, thresholds: {}'.format(
            i, metric, np.array(thresholds)))

    return metric, thresholds


def _get_metric(results, metric_type):
    if metric_type == 'f1':
        return results['overall']['f_measure']['f_measure']
    elif metric_type == 'er':
        return results['overall']['error_rate']['error_rate']
    elif metric_type == 'precision':
        return results['overall']['f_measure']['precision']
    elif metric_type == 'recall':
        return results['overall']['f_measure']['recall']


def calculate_sed_gradient(output_dict, submission_path, reference_csv_path, 
    sed_params_dict, metric_type):
    """Optimize thresholds for SED.

    Args:
      output_dict: {'clipwise_output': (N, classes_num), 
                    'framewise_output': (N, frames_num, classes_num)}
      submission_path: str
      reference_csv_path: str
      sed_params_dict: dict
      metric_type: 'f1' | 'er'

    Returns:
      grads: vector
    """
    predict_event_list = frame_prediction_to_event_prediction(
        output_dict, sed_params_dict)

    write_submission(predict_event_list, submission_path)
    results = official_evaluate(reference_csv_path, submission_path)
    value = _get_metric(results, metric_type)
    
    grads = []
    params = sed_dict_to_params(sed_params_dict)
 
    for k, param in enumerate(params):
        print('Param index: {} / {}'.format(k, len(params)))
        new_params = params.copy()
        delta = 0.1
        cnt = 0
        while cnt < 3:
            cnt += 1
            new_params[k] += delta
            new_params_dict = sed_params_to_dict(new_params, sed_params_dict)

            predict_event_list = frame_prediction_to_event_prediction(
                output_dict, new_params_dict)

            write_submission(predict_event_list, submission_path)
            results = official_evaluate(reference_csv_path, submission_path)
            new_value = _get_metric(results, metric_type)

            if new_value != value:
                break

        grad = (new_value - value) / (delta * cnt)
        grads.append(grad)

    return grads
    

def sed_dict_to_params(sed_params_dict):
    keys = ['audio_tagging_threshold', 'sed_high_threshold', 'sed_low_threshold']
    params = []
    for key in keys:
        params += sed_params_dict[key]
    return params


def sed_params_to_dict(params, sed_params_dict): 
    classes_num = config.classes_num
    new_sed_params_dict = {'audio_tagging_threshold': params[0 : 17], 
        'sed_high_threshold': params[17 : 34], 
        'sed_low_threshold': params[34 : 51], 
        'n_smooth': sed_params_dict['n_smooth'], 
        'n_salt': sed_params_dict['n_salt']}
    return new_sed_params_dict


def optimize_sed_with_gd(output_dict, submission_path, reference_csv_path, 
    sed_params_dict, metric_type):
    """Optimize thresholds for SED.

    Args:
      output_dict: {'clipwise_output': (N, classes_num), 
                    'framewise_output': (N, frames_num, classes_num)}
      submission_path: str
      reference_csv_path: str
      sed_params_dict: dict
      metric_type: 'f1' | 'er'

    Returns:
      metric: float
      sed_params_dict: dict, optimized thresholds
    """
    predict_event_list = frame_prediction_to_event_prediction(
        output_dict, sed_params_dict)

    write_submission(predict_event_list, submission_path)
    results = official_evaluate(reference_csv_path, submission_path)
    metric = _get_metric(results, metric_type)
    print('Initial {}: {}'.format(metric_type, metric))
    print('Running optimization on thresholds.')

    opt = Adam()
    opt.alpha = 2e-2
    for i in range(10):
        grads = calculate_sed_gradient(output_dict, submission_path, 
            reference_csv_path, sed_params_dict, metric_type)

        if metric_type == 'f1':
            grads = [-e for e in grads]
        elif metric_type == 'er':
            pass

        params = sed_dict_to_params(sed_params_dict)
        sed_params = opt.GetNewParams(params, grads)
        sed_params_dict = sed_params_to_dict(sed_params, sed_params_dict)
        
        predict_event_list = frame_prediction_to_event_prediction(output_dict, 
            sed_params_dict)

        write_submission(predict_event_list, submission_path)
        results = official_evaluate(reference_csv_path, submission_path)
        metric = _get_metric(results, metric_type)
        print('******')
        print('Iteration: {}, {}: {}'.format(i, metric_type, metric))

    return metric, sed_params_dict


def calculate_precision_recall(y_true, output, thresholds, average):
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
        return precision, recall
    
    else:
        raise Exception('Incorrect argument!')



def optimize_thresholds(args):
    """Optimize thresholds for AT and SED. The thresholds are used to obtain 
    eventwise predictions from framewise predictions. 
    """

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    few_shots = args.few_shots
    random_seed = args.random_seed
    iteration = args.iteration
    filename = args.filename
    mini_data = False
    pretrain = False
    
    classes_num = config.classes_num
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
     
    test_hdf5_path = os.path.join(workspace, 'features', 
        'testing.waveform.h5'.format(prefix))

    evaluate_hdf5_path = os.path.join(workspace, 'features', 
        'evaluation.waveform.h5'.format(prefix))

    test_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_testing_set.csv')
        
    evaluate_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_evaluation_set.csv')

    predictions_dir = os.path.join(workspace, 'predictions', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'pretrain={}'.format(pretrain), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'few_shots={}'.format(few_shots), 
        'random_seed={}'.format(random_seed), 'freeze_base={}'.format(freeze_base),
        'batch_size={}'.format(batch_size))
    
    tmp_submission_path = os.path.join(workspace, '_tmp_submission', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'pretrain={}'.format(pretrain), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'few_shots={}'.format(few_shots), 'random_seed={}'.format(random_seed), 
        'freeze_base={}'.format(freeze_base), '_submission.csv')
    create_folder(os.path.dirname(tmp_submission_path))

    post_processing_params_dir = os.path.join(workspace, 'post_processing_params', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'pretrain={}'.format(pretrain), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'few_shots={}'.format(few_shots), 'random_seed={}'.format(random_seed), 
        'freeze_base={}'.format(freeze_base))
    create_folder(post_processing_params_dir)
    
    # Optimize thresholds for AT
    if True:
        average = 'micro'   # micro F value is used in DCASE 2017 Task 4 evaluation

        # Paths
        data_type = 'test'  # Optimize thresholds on test (validation) set.
        prediction_path = os.path.join(predictions_dir, 
            '{}_iterations.prediction.{}.h5'.format(iteration, data_type))

        # Load ground truth weak target
        with h5py.File(test_hdf5_path, 'r') as hf:
            audio_name = [name.decode() for name in hf['audio_name'][:]]
            weak_target = hf['weak_target'][:].astype(np.float32)

        # Load prediction
        (clipwise_prediction, framewise_prediction) = load_prediction(prediction_path)
        
        manual_thres_f1 = calculate_f1(weak_target, clipwise_prediction, 
            thresholds=[0.3] * classes_num, average=average)
        
        # Optimize thresholds
        (auto_thres_f1, auto_thresholds) = optimize_at_with_gd(weak_target, 
            clipwise_prediction, [0.3] * classes_num, average=average)

        print('test manual_thres f1: {}'.format(manual_thres_f1))
        print('test auto_thres f1: {}'.format(auto_thres_f1))

        post_processing_params_path = os.path.join(post_processing_params_dir, 
            'at_f1.npy')
        cPickle.dump(auto_thresholds, open(post_processing_params_path, 'wb'))
        print('Write post_processing_params to {}'.format(post_processing_params_path))

    # Optimize thresholds for SED
    if True:
        # Initial thresholds for SED
        sed_params_dict = {
            'audio_tagging_threshold': [0.3] * classes_num, 
            'sed_high_threshold': [0.3] * classes_num, 
            'sed_low_threshold': [0.05] * classes_num, 
            'n_smooth': [1] * classes_num, 
            'n_salt': [1] * classes_num}

        # Paths
        prediction_path = os.path.join(predictions_dir, 
            '{}_iterations.prediction.test.h5'.format(iteration))

        # Load SED prediction
        (clipwise_prediction, framewise_prediction) = load_prediction(prediction_path)
        output_dict = {'audio_name': audio_name, 
            'clipwise_output': clipwise_prediction, 
            'framewise_output': framewise_prediction}

        for metric_type in ['f1', 'er']:
            # Optimize thresholds for SED
            (auto_thres_f1, auto_thresholds) = optimize_sed_with_gd(
                output_dict, tmp_submission_path, test_reference_csv_path, 
                sed_params_dict, metric_type=metric_type)

            post_processing_params_path = os.path.join(post_processing_params_dir, 
                'sed_{}.npy'.format(metric_type))
            cPickle.dump(auto_thresholds, open(post_processing_params_path, 'wb'))
            print('Write post_processing_params to {}'.format(post_processing_params_path))


def calculate_metrics(args):
    """Calculate metrics with optimized thresholds
    """
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    few_shots = args.few_shots
    random_seed = args.random_seed
    iteration = args.iteration
    filename = args.filename
    mini_data = False
    pretrain = False
    
    classes_num = config.classes_num
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
     
    test_hdf5_path = os.path.join(workspace, 'features', 
        'testing.waveform.h5'.format(prefix))

    evaluate_hdf5_path = os.path.join(workspace, 'features', 
        'evaluation.waveform.h5'.format(prefix))

    test_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_testing_set.csv')
        
    evaluate_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_evaluation_set.csv')

    predictions_dir = os.path.join(workspace, 'predictions', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'pretrain={}'.format(pretrain), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'few_shots={}'.format(few_shots), 
        'random_seed={}'.format(random_seed), 'freeze_base={}'.format(freeze_base),
        'batch_size={}'.format(batch_size))
    
    tmp_submission_path = os.path.join(workspace, '_tmp_submission', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'pretrain={}'.format(pretrain), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'few_shots={}'.format(few_shots), 'random_seed={}'.format(random_seed), 
        'freeze_base={}'.format(freeze_base), '_submission.csv')

    post_processing_params_dir = os.path.join(workspace, 'post_processing_params', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'pretrain={}'.format(pretrain), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'few_shots={}'.format(few_shots), 'random_seed={}'.format(random_seed), 
        'freeze_base={}'.format(freeze_base))
    
    t1 = time.time()

    # Calculate metrics for AT
    if True:
        print('------ AT ------')

        # Load auto thresholds
        post_processing_params_path = os.path.join(post_processing_params_dir, 
            'at_f1.npy')
        auto_thresholds = cPickle.load(open(post_processing_params_path, 'rb'))

        average = 'micro'

        # ------ Test metrics ------
        # Paths
        prediction_path = os.path.join(predictions_dir, 
            '{}_iterations.prediction.test.h5'.format(iteration))

        # Load ground truth weak target
        with h5py.File(test_hdf5_path, 'r') as hf:
            weak_target = hf['weak_target'][:].astype(np.float32)

        # Load prediction probability
        (clipwise_prediction, framewise_prediction) = load_prediction(prediction_path)

        # Macro mAP
        mAP = metrics.average_precision_score(weak_target, clipwise_prediction, 
            average='macro')
        print('test macro mAP: {:.3f}'.format(mAP))

        # Metrics without thresholds optimization
        manual_thres_f1 = calculate_f1(weak_target, clipwise_prediction, 
            thresholds=[0.3] * classes_num, average=average)

        manual_thres_prec, manual_thres_recall = calculate_precision_recall(
            weak_target, clipwise_prediction, thresholds=[0.3] * classes_num, 
            average=average)

        print('(no_opt_thres) test f1: {:.3f}, prec: {:.3f}, recall: {:.3f}'.format(
            manual_thres_f1, manual_thres_prec, manual_thres_recall))

        # Metrics with thresholds optimization
        auto_thres_f1 = calculate_f1(weak_target, clipwise_prediction, 
            thresholds=auto_thresholds, average=average)
        auto_thres_prec, auto_thres_recall = calculate_precision_recall(
            weak_target, clipwise_prediction, thresholds=auto_thresholds, 
            average=average)

        print('(opt_thres)    test f1: {:.3f}, prec: {:.3f}, recall: {:.3f}'.format(
            auto_thres_f1, auto_thres_prec, auto_thres_recall))

        # ------ Evaluate metrics ------
        # Paths
        prediction_path = os.path.join(predictions_dir, 
            '{}_iterations.prediction.{}.h5'.format(iteration, 'evaluate'))

        # Load ground truth weak target
        with h5py.File(evaluate_hdf5_path, 'r') as hf:
            weak_target = hf['weak_target'][:].astype(np.float32)

        # Load prediction probability
        (clipwise_prediction, framewise_prediction) = load_prediction(prediction_path)

        # Macro mAP
        mAP = metrics.average_precision_score(weak_target, clipwise_prediction, 
            average='macro')
        print('evaluate macro mAP: {:.3f}'.format(mAP))

        # Metrics without thresholds optimization
        manual_thres_f1 = calculate_f1(weak_target, clipwise_prediction, 
            thresholds=[0.3] * classes_num, average=average)
        manual_thres_prec, manual_thres_recall = calculate_precision_recall(
            weak_target, clipwise_prediction, thresholds=[0.3] * classes_num, 
            average=average)

        print('(no_opt_thres) evaluate f1: {:.3f}, prec: {:.3f}, recall: {:.3f}'.format(
            manual_thres_f1, manual_thres_prec, manual_thres_recall))

        # Metrics with thresholds optimization
        auto_thres_f1 = calculate_f1(weak_target, clipwise_prediction, 
            auto_thresholds, average=average)
        auto_thres_prec, auto_thres_recall = calculate_precision_recall(
            weak_target, clipwise_prediction, thresholds=auto_thresholds, 
            average=average)
        
        print('(opt_thres)    evaluate f1: {:.3f}, prec: {:.3f}, recall: {:.3f}'.format(
            auto_thres_f1, auto_thres_prec, auto_thres_recall))
        print()
 
    # Calculate metrics for SED
    if True:
        print('------ SED ------')

        # Initial thresholds for SED
        sed_params_dict = {
            'audio_tagging_threshold': [0.3] * classes_num, 
            'sed_high_threshold': [0.3] * classes_num, 
            'sed_low_threshold': [0.05] * classes_num, 
            'n_smooth': [1] * classes_num, 
            'n_salt': [1] * classes_num}

        for metric_idx, metric_type in enumerate(['f1', 'er']):
            print('*** Metric type: {} ***'.format(metric_type))
            
            # Load optimized thresholds
            post_processing_params_path = os.path.join(post_processing_params_dir, 
                'sed_{}.npy'.format(metric_type))

            auto_sed_params_dict = cPickle.load(open(post_processing_params_path, 'rb'))

            # ------ Test ------
            # Paths
            prediction_path = os.path.join(predictions_dir, 
                '{}_iterations.prediction.test.h5'.format(iteration))

            # Load ground truth strong target
            with h5py.File(test_hdf5_path, 'r') as hf:
                audio_name = [name.decode() for name in hf['audio_name'][:]]
                strong_target = hf['strong_target'][:].astype(np.float32)

            # Load prediction probability
            (clipwise_prediction, framewise_prediction) = load_prediction(prediction_path)

            output_dict = {'audio_name': audio_name, 
                'clipwise_output': clipwise_prediction, 
                'framewise_output': framewise_prediction}

            # Macro framewise mAP
            if metric_idx == 0:
                mAP = metrics.average_precision_score(
                    strong_target.reshape((strong_target.shape[0]*strong_target.shape[1], strong_target.shape[2])), 
                    framewise_prediction.reshape((framewise_prediction.shape[0]*framewise_prediction.shape[1], framewise_prediction.shape[2])), 
                    average='macro')

                print('test macro mAP: {:.3f}'.format(mAP))

            # Eventwise prediction without thresholds optimization
            predict_event_list = frame_prediction_to_event_prediction(
                output_dict, sed_params_dict)
            write_submission(predict_event_list, tmp_submission_path)
            results = official_evaluate(test_reference_csv_path, tmp_submission_path)
            
            metric = _get_metric(results, metric_type)
            print('(no_opt_thres) test {}: {:.3f}'.format(metric_type, metric))
        
            # Eventwise prediction with thresholds optimization
            predict_event_list = frame_prediction_to_event_prediction(
                output_dict, auto_sed_params_dict)
            write_submission(predict_event_list, tmp_submission_path)
            results = official_evaluate(test_reference_csv_path, tmp_submission_path)
            metric = _get_metric(results, metric_type)
            print('(opt_thres)    test {}: {:.3f}'.format(metric_type, metric))
        
            # ------ Evaluate ------
            # Paths
            prediction_path = os.path.join(predictions_dir, 
                '{}_iterations.prediction.evaluate.h5'.format(iteration))

            # Load ground truth strong target
            with h5py.File(evaluate_hdf5_path, 'r') as hf:
                audio_name = [name.decode() for name in hf['audio_name'][:]]
                strong_target = hf['strong_target'][:].astype(np.float32)

            # Load prediction probability
            (clipwise_prediction, framewise_prediction) = load_prediction(prediction_path)

            output_dict = {'audio_name': audio_name, 
                'clipwise_output': clipwise_prediction, 
                'framewise_output': framewise_prediction}

            # Macro framewise mAP
            if metric_idx == 0:
                mAP = metrics.average_precision_score(
                    strong_target.reshape((strong_target.shape[0]*strong_target.shape[1], strong_target.shape[2])), 
                    framewise_prediction.reshape((framewise_prediction.shape[0]*framewise_prediction.shape[1], framewise_prediction.shape[2])), 
                    average='macro')

                print('evaluate mAP: {:.3f}'.format(mAP))
            
            # Eventwise prediction without thresholds optimization
            predict_event_list = frame_prediction_to_event_prediction(
                output_dict, sed_params_dict)
            write_submission(predict_event_list, tmp_submission_path)
            results = official_evaluate(evaluate_reference_csv_path, 
                tmp_submission_path)
            value = _get_metric(results, metric_type)
            print('(no_opt_thres) evaluate {}: {:.3f}'.format(metric_type, value))
        
            # Metrics with thresholds optimization
            predict_event_list = frame_prediction_to_event_prediction(
                output_dict, auto_sed_params_dict)
            write_submission(predict_event_list, tmp_submission_path)
            results = official_evaluate(evaluate_reference_csv_path, 
                tmp_submission_path)
            value = _get_metric(results, metric_type)
            print('(opt_thres)    evaluate {}: {:.3f}'.format(metric_type, value))
            print()
        
        print('time: {:.3f} s'.format(time.time() - t1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_optimize_thresholds = subparsers.add_parser('optimize_thresholds')
    parser_optimize_thresholds.add_argument('--dataset_dir', type=str, required=True)
    parser_optimize_thresholds.add_argument('--workspace', type=str, required=True)
    parser_optimize_thresholds.add_argument('--holdout_fold', type=str, choices=['1', 'none'], required=True)
    parser_optimize_thresholds.add_argument('--model_type', type=str, required=True)
    parser_optimize_thresholds.add_argument('--freeze_base', action='store_true', default=False)
    parser_optimize_thresholds.add_argument('--loss_type', type=str, required=True)
    parser_optimize_thresholds.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_optimize_thresholds.add_argument('--learning_rate', type=float, required=True)
    parser_optimize_thresholds.add_argument('--batch_size', type=int, required=True)
    parser_optimize_thresholds.add_argument('--few_shots', type=int, required=True)
    parser_optimize_thresholds.add_argument('--random_seed', type=int, required=True)
    parser_optimize_thresholds.add_argument('--iteration', type=int, required=True)
     
    parser_calculate_metrics = subparsers.add_parser('calculate_metrics')
    parser_calculate_metrics.add_argument('--dataset_dir', type=str, required=True)
    parser_calculate_metrics.add_argument('--workspace', type=str, required=True)
    parser_calculate_metrics.add_argument('--holdout_fold', type=str, choices=['1', 'none'], required=True)
    parser_calculate_metrics.add_argument('--model_type', type=str, required=True)
    parser_calculate_metrics.add_argument('--freeze_base', action='store_true', default=False)
    parser_calculate_metrics.add_argument('--loss_type', type=str, required=True)
    parser_calculate_metrics.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_calculate_metrics.add_argument('--learning_rate', type=float, required=True)
    parser_calculate_metrics.add_argument('--batch_size', type=int, required=True)
    parser_calculate_metrics.add_argument('--few_shots', type=int, required=True)
    parser_calculate_metrics.add_argument('--random_seed', type=int, required=True)
    parser_calculate_metrics.add_argument('--iteration', type=int, required=True)

    args = parser.parse_args()
    args.filename = 'pytorch_main'

    if args.mode == 'optimize_thresholds':
        optimize_thresholds(args)
         
    elif args.mode == 'calculate_metrics':
        calculate_metrics(args)

    else:
        raise Exception('Error argument!')
