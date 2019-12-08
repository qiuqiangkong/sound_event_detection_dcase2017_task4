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


def inference(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    iteration = args.iteration
    cuda = args.cuda and torch.cuda.is_available()
    select = args.select
    mini_data = args.mini_data
    filename = args.filename
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    labels = config.labels
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    test_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'testing.h5')
        
    evaluate_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'evaluation.h5')
        
    test_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_testing_set.csv')
        
    evaluate_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_evaluation_set.csv')
        
    scalar_path = os.path.join(workspace, 'scalars', 
        'logmel_{}frames_{}melbins'.format(frames_per_second, mel_bins), 
        'training.h5')
        
    checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'holdout_fold={}'.format(holdout_fold), model_type, loss_type, 
        'balanced={}'.format(balanced), 'augmentation={}'.format(augmentation), 
        'accumulation_steps={}'.format(accumulation_steps), 
        '{}_iterations.pth'.format(iteration))

    figs_dir = 'results_fig_for_paper'
    create_folder(figs_dir)

    # Load scalar
    scalar = load_scalar(scalar_path)
    
    # Model
    Model = eval(model_type)
    model = Model(classes_num)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    if cuda:
        model.cuda()
    
    data_generator = Base(
        train_hdf5_path=None, 
        test_hdf5_path=test_hdf5_path, 
        evaluate_hdf5_path=evaluate_hdf5_path, 
        holdout_fold=holdout_fold, 
        scalar=scalar, 
        batch_size=batch_size, 
        seed=1234)

    result = {}

    # Calculate statistics
    for data_type in ['test', 'evaluate']:
        generate_func = data_generator.generate_validate(
            data_type=data_type, 
            max_iteration=None, 
            shuffle=False)
            
        # Forward
        output_dict = forward(
            model=model, 
            generate_func=generate_func, 
            cuda=cuda, 
            return_input=True, 
            return_target=True)

        feature = output_dict['feature']
        clipwise_output = output_dict['clipwise_output']
        framewise_output = output_dict['framewise_output']
        weak_target = output_dict['weak_target']
        strong_target = output_dict['strong_target']
        
        # AT & segment mAP
        average_precision = metrics.average_precision_score(weak_target, clipwise_output, average=None)
        strong_average_precision = sed_average_precision(strong_target, framewise_output, average=None)

        print('Clipwise mAP: {}'.format(np.mean(average_precision)))
        print('Framewise mAP: {}'.format(np.mean(strong_average_precision)))

        result[data_type] = {
            'clipwise_output': clipwise_output, 
            'framewise_output': framewise_output, 
            'weak_target': weak_target, 
            'strong_target': strong_target
        }

        if True:
            print('--- {} ---'.format(data_type))
            # AT manual & auto threshold F value
            manual_thres_f1 = f1_score(weak_target, clipwise_output, thresholds=[0.3] * classes_num)

            if data_type == 'test':
                (auto_thres_f1, auto_thresholds) = get_best_f1_score(weak_target, clipwise_output)
            elif data_type == 'evaluate':
                auto_thres_f1 = f1_score(weak_target, clipwise_output, auto_thresholds)

            print('Manual threshold AT f1: {:.3f}'.format(manual_thres_f1))
            print('Auto threshold AT f1: {:.3f}'.format(auto_thres_f1))
            print('Auto thresholds: {}'.format(auto_thresholds))
            
            # Framewise manual & auto threshold F value
            manual_thres_sed_f1 = f1_score(strong_target, framewise_output, thresholds=[0.3] * classes_num)

            if data_type == 'test':
                (auto_thres_sed_f1, auto_thresholds_sed) = get_best_f1_score(strong_target[0::10], framewise_output[0::10])
            elif data_type == 'evaluate':
                auto_thres_sed_f1 = f1_score(strong_target, framewise_output, auto_thresholds_sed)

            print('Manual threshold SED f1: {:.3f}'.format(manual_thres_sed_f1))
            print('Auto threshold SED f1: {:.3f}'.format(auto_thres_sed_f1))
            print('Auto thresholds SED: {}'.format(auto_thresholds_sed))

        #
        # sed_params_dict = {
        #     'audio_tagging_threshold': 0.3, 
        #     'sed_high_threshold': 0.5, 
        #     'sed_low_threshold': 0.2, 
        #     'n_smooth': 10, 
        #     'n_salt': 10}

            sed_params_dict = {'audio_tagging_threshold': 0.5, 'sed_high_threshold': 0.5, 'sed_low_threshold': 0.2, 'n_smooth': 1, 'n_salt': 1}
            #sed_params_dict = {'audio_tagging_threshold': 0.5, 'sed_high_threshold': 0.5, 'sed_low_threshold': 0.2, 'n_smooth': 10, 'n_salt': 10}
            #sed_params_dict = {'audio_tagging_threshold': auto_thresholds, 'sed_high_threshold': 0.5, 'sed_low_threshold': 0.2, 'n_smooth': 10, 'n_salt': 10}


            if data_type == 'test':
                reference_csv_path = test_reference_csv_path
            elif data_type == 'evaluate':
                reference_csv_path = evaluate_reference_csv_path

            submission_path = '_tmp_submission.csv'
            predict_event_list = frame_prediction_to_event_prediction(output_dict, sed_params_dict)
            write_submission(predict_event_list, submission_path)
            results = official_evaluate(reference_csv_path, submission_path)

            if data_type == 'test':
                (auto_thres_sed_result, auto_threshods) = get_best_sed(output_dict, sed_params_dict, submission_path, reference_csv_path)
            elif data_type == 'evaluate':
                sed_params_dict['audio_tagging_threshold'] = auto_threshods
                predict_event_list = frame_prediction_to_event_prediction(output_dict, sed_params_dict)
                write_submission(predict_event_list, submission_path)
                auto_thres_sed_result = official_evaluate(reference_csv_path, submission_path)

            print('Manual threshold SED er: {:.3f}'.format(results['overall']['error_rate']['error_rate']))
            print('Auto threshold SED er: {:.3f}'.format(auto_thres_sed_result['overall']['error_rate']['error_rate']))
            print('Auto thresholds: {}'.format(auto_threshods))


            tmp = [auto_thres_sed_result['class_wise'][label]['error_rate'] for label in labels]

    # Plot
    def _get_threshold_index(thresholds):
        theta = 0
        indexes = []
        for j, threshold in enumerate(thresholds):
            if threshold >= theta:
                indexes.append(j)
                theta += 0.02
        return indexes

    # Plot prediction curve
    if select == '0':
        for n in range(200, strong_target.shape[0]):
            fig, axs = plt.subplots(4, 5, figsize=(12, 8))
            for k in range(classes_num):
                axs[k // 5, k % 5].plot(framewise_output[n, :, k], c='b')
                # axs[k // 5, k % 5].plot(debug1[n, :, k], c='b')
                axs[k // 5, k % 5].plot(np.clip(strong_target[n, :, k], 0.01, 0.99), c='r')
                axs[k // 5, k % 5].set_ylim(0, 1)
                axs[k // 5, k % 5].set_title('{}, {:.3f}'.format(labels[k][0:15], clipwise_output[n, k]), fontsize=8)
            # axs[3, 2].matshow(tmp[n][0:64], origin='lower', aspect='auto', cmap='jet')
            axs[3, 3].matshow(output_dict['embedding'][n], origin='lower', aspect='auto', cmap='jet')
            axs[3, 4].matshow(feature[n].T, origin='lower', aspect='auto', cmap='jet')
            plt.tight_layout(pad=0, w_pad=0, h_pad=0)

            fig_path = os.path.join(figs_dir, 'sed_visualize', '{}.pdf'.format(n))
            create_folder(os.path.dirname(fig_path))
            plt.savefig(fig_path)
            print('Save figure to {}'.format(fig_path))
            if n == 205:
                break
        
    # Plot thresholds of sound classes
    if select == '2':
        
        fig, axs = plt.subplots(3, 6, sharex=True, figsize=(12, 5.5))
        colors = ['b', 'r']
        for m, data_type in enumerate(['test', 'evaluate']):
            for k in range(classes_num):
                (prec, recall, thresholds) = metrics.precision_recall_curve(result[data_type]['weak_target'][:, k], result[data_type]['clipwise_output'][:, k])
                indexes = _get_threshold_index(thresholds)
                prec = prec[indexes]
                recall = recall[indexes]
                thresholds = thresholds[indexes]

                for i1 in range(len(thresholds)):
                    axs[k // 6, k % 6].scatter(recall[i1], prec[i1], s=10, c=colors[m], marker='+', label='first', alpha=thresholds[i1])

                axs[k // 6, k % 6].plot(recall, prec, c='grey', linewidth=0.8, alpha=0.5)
                axs[k // 6, k % 6].set_title(labels[k], fontsize=8)
                axs[k // 6, k % 6].set_xlim(0, 1)
                axs[k // 6, k % 6].set_ylim(0, 1.01)
                axs[k // 6, k % 6].set_xlabel('Recall', fontsize=8)
                axs[k // 6, k % 6].set_ylabel('Precision', fontsize=8)

            axs[2, 5].set_visible(False)
            #
            ax = fig.add_axes([0.88, 0.095, 0.008, 0.23])
            cmap = mpl.cm.Reds
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')

            # 
            ax = fig.add_axes([0.94, 0.095, 0.008, 0.23])
            cmap = mpl.cm.Blues
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')

        plt.tight_layout()

        fig_path = os.path.join(figs_dir, 'at_thresholds.pdf')
        plt.savefig(fig_path)
        print('Save figure to {}'.format(fig_path))

    if select == '2b':
        
        fig, axs = plt.subplots(4, 5, sharex=True, figsize=(12, 8))
        colors = ['b', 'r']
        for m, data_type in enumerate(['test', 'evaluate']):
            for k in range(classes_num):
                (prec, recall, thresholds) = metrics.precision_recall_curve(result[data_type]['strong_target'][:, :, k].flatten(), result[data_type]['framewise_output'][:, :, k].flatten())
                indexes = _get_threshold_index(thresholds)
                prec = prec[indexes]
                recall = recall[indexes]
                thresholds = thresholds[indexes]

                for i1 in range(len(thresholds)):
                    axs[k // 5, k % 5].scatter(recall[i1], prec[i1], s=10, c=colors[m], marker='+', label='first', alpha=thresholds[i1])
                axs[k // 5, k % 5].plot(recall, prec, c='k', linewidth=0.5, alpha=0.2)
        
                axs[k // 5, k % 5].set_title(labels[k])
                axs[k // 5, k % 5].set_xlim(0, 1)
                axs[k // 5, k % 5].set_ylim(0, 1)

        fig_path = os.path.join(figs_dir, 'at_thresholds_b.pdf')
        plt.savefig(fig_path)
        print('Save figure to {}'.format(fig_path))
            
    # Visualize embedding
    if select == 3:
        for batch_data_dict in generate_func:
    
            # Predict
            batch_feature = move_data_to_gpu(batch_data_dict['feature'], cuda)
            
            
            with torch.no_grad():
                model.eval()
                
                (batch_output, batch_embedding) = model(batch_feature, return_embedding=True)
                batch_target = batch_data_dict['target']
                batch_name = batch_data_dict['audio_name']
        
                m1 = 12
                print(batch_name[m1])
                print(batch_target[m1])
                print(np.around(batch_output[m1], decimals=2))
                fig, axs = plt.subplots(1,3, figsize=(15, 8))
                feature = batch_feature.data.cpu().numpy()[m1]
                embedding = batch_embedding.data.cpu().numpy()[m1]
                embedding = np.mean(embedding, axis=-1)
                mask = np.zeros_like(embedding)
                mask[np.arange(embedding.shape[0]), np.argmax(embedding, axis=-1)] = 1
                axs[0].matshow(feature.T, origin='lower', aspect='auto', cmap='jet')
                axs[1].matshow(embedding, origin='lower', aspect='auto', cmap='jet')
                axs[2].matshow(mask, origin='lower', aspect='auto', cmap='jet')
                plt.savefig('_tmp.png')
                import crash
                asdf

                
    # Visualize feature map
    if select == 4:
        for batch_data_dict in generate_func:
    
            # Predict
            batch_feature = move_data_to_gpu(batch_data_dict['feature'], cuda)
            
            
            with torch.no_grad():
                model.eval()
                
                (batch_output, batch_embedding) = model(batch_feature, return_embedding=True)
                batch_target = batch_data_dict['target']
                batch_name = batch_data_dict['audio_name']
        
                m1 = 1
                print(batch_name[m1])
                print(batch_target[m1])
                print(np.around(batch_output[m1], decimals=2))
                fig, axs = plt.subplots(3, 3, figsize=(15, 8))
                feature = batch_feature.data.cpu().numpy()[m1]
                embedding = batch_embedding.data.cpu().numpy()[m1]
                axs[0, 0].matshow(feature.T, origin='lower', aspect='auto', cmap='jet')
                axs[0, 1].matshow(embedding[0].T, origin='lower', aspect='auto', cmap='jet')
                axs[0, 2].matshow(embedding[1].T, origin='lower', aspect='auto', cmap='jet')
                axs[1, 0].matshow(embedding[2].T, origin='lower', aspect='auto', cmap='jet')
                axs[1, 1].matshow(embedding[3].T, origin='lower', aspect='auto', cmap='jet')
                axs[1, 2].matshow(embedding[4].T, origin='lower', aspect='auto', cmap='jet')
                axs[2, 0].matshow(embedding[5].T, origin='lower', aspect='auto', cmap='jet')
                axs[2, 1].matshow(embedding[6].T, origin='lower', aspect='auto', cmap='jet')
                axs[2, 2].matshow(embedding[7].T, origin='lower', aspect='auto', cmap='jet')
                plt.savefig('_tmp.png')

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
