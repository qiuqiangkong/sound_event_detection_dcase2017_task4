import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import pickle
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
from evaluate import Evaluator
from config import (sample_rate, classes_num, mel_bins, fmin, fmax, 
    window_size, hop_size, window, pad_mode, center, device, ref, amin, top_db)
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup
from utilities import (create_folder, get_filename, create_logging, 
    StatisticsContainer, Mixup)
from data_generator import (DCASE2017Task4Dataset, TrainSampler, TestSampler, 
    collate_fn)
from models import * 


def train(args):
    """Train and evaluate.

    Args:
      dataset_dir: str
      workspace: str
      holdout_fold: '1'
      model_type: str, e.g., 'Cnn_9layers_Gru_FrameAtt'
      loss_type: str, e.g., 'clip_bce'
      augmentation: str, e.g., 'mixup'
      learning_rate, float
      batch_size: int
      resume_iteration: int
      stop_iteration: int
      device: 'cuda' | 'cpu'
      mini_data: bool
    """

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    resume_iteration = args.resume_iteration
    stop_iteration = args.stop_iteration
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    mini_data = args.mini_data
    filename = args.filename

    loss_func = get_loss_func(loss_type)
    num_workers = 8
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    train_hdf5_path = os.path.join(workspace, 'hdf5s', 
        '{}training.h5'.format(prefix))

    test_hdf5_path = os.path.join(workspace, 'hdf5s', 
        '{}testing.h5'.format(prefix))

    evaluate_hdf5_path = os.path.join(workspace, 'hdf5s', 
        'evaluation.h5'.format(prefix))

    test_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_testing_set.csv')
        
    evaluate_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_evaluation_set.csv')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)

    tmp_submission_path = os.path.join(workspace, '_tmp_submission', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '_submission.csv')
    create_folder(os.path.dirname(tmp_submission_path))

    statistics_path = os.path.join(workspace, 'statistics', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pickle')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', '{}{}'.format(prefix, filename), 
        'holdout_fold={}'.format(holdout_fold), 'model_type={}'.format(model_type), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
        'batch_size={}'.format(batch_size))
    create_logging(logs_dir, 'w')
    logging.info(args)

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
    
    # Model
    assert model_type, 'Please specify model_type!'
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
        classes_num)

    if resume_iteration:
        resume_checkpoint_path = os.path.join(checkpoints_dir, 
            '{}_iterations.pth'.format(resume_iteration))
        logging.info('Load resume model from {}'.format(resume_checkpoint_path))
        resume_checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(resume_checkpoint['model'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = resume_checkpoint['iteration']
    else:
        iteration = 0

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in device:
        model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    # Dataset
    dataset = DCASE2017Task4Dataset()

    # Sampler
    train_sampler = TrainSampler(
        hdf5_path=train_hdf5_path, 
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size)
    
    test_sampler = TestSampler(hdf5_path=test_hdf5_path, batch_size=batch_size)

    evaluate_sampler = TestSampler(hdf5_path=evaluate_hdf5_path, batch_size=batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    evaluate_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=evaluate_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)
    
    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)
    
    # Evaluator
    evaluator = Evaluator(model=model)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    train_bgn_time = time.time()
    
    # Train on mini batches
    for batch_data_dict in train_loader:

        # Evaluate
        if (iteration % 1000 == 0 and iteration > resume_iteration):# or (iteration == 0):
        
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()

            for (data_type, data_loader, reference_csv_path) in [
                ('test', test_loader, test_reference_csv_path), 
                ('evaluate', evaluate_loader, evaluate_reference_csv_path)]:

                # Calculate tatistics
                (statistics, _) = evaluator.evaluate(
                    data_loader, reference_csv_path, tmp_submission_path)

                logging.info('{} statistics:'.format(data_type))
                logging.info('    Clipwise mAP: {:.3f}'.format(np.mean(statistics['clipwise_ap'])))
                logging.info('    Framewise mAP: {:.3f}'.format(np.mean(statistics['framewise_ap'])))
                logging.info('    {}'.format(statistics['sed_metrics']['overall']['error_rate']))

                statistics_container.append(data_type, iteration, statistics)

            statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()

        # Save model 
        if iteration % 10000 == 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                'optimizer': optimizer.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        
        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(batch_data_dict['waveform']))

        # Move data to GPU
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        # Train
        model.train()

        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['waveform'], batch_data_dict['mixup_lambda'])
            batch_target_dict = {'target': do_mixup(batch_data_dict['target'], batch_data_dict['mixup_lambda'])}
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None)
            batch_target_dict = {'target': batch_data_dict['target']}

        # loss
        loss = loss_func(batch_output_dict, batch_target_dict)
        print(iteration, loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == stop_iteration:
            break 
            
        iteration += 1
        

def inference_prob(self):
    """Inference test and evaluate data and dump predicted probabilites to 
    pickle files.

    Args:
      dataset_dir: str
      workspace: str
      holdout_fold: '1'
      model_type: str, e.g., 'Cnn_9layers_Gru_FrameAtt'
      loss_type: str, e.g., 'clip_bce'
      augmentation: str, e.g., 'mixup'
      batch_size: int
      device: 'cuda' | 'cpu'
    """
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    batch_size = args.batch_size
    iteration = args.iteration
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename

    num_workers = 8

    # Paths
    test_hdf5_path = os.path.join(workspace, 'hdf5s', 'testing.h5')

    evaluate_hdf5_path = os.path.join(workspace, 'hdf5s', 'evaluation.h5')

    test_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_testing_set.csv')
        
    evaluate_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_evaluation_set.csv')

    checkpoint_path = os.path.join(workspace, 'checkpoints', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '{}_iterations.pth'.format(iteration))

    predictions_dir = os.path.join(workspace, 'predictions', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(predictions_dir)

    tmp_submission_path = os.path.join(workspace, '_tmp_submission', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '_submission.csv')
    create_folder(os.path.dirname(tmp_submission_path))

    # Load model
    assert model_type, 'Please specify model_type!'
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
        classes_num)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in device:
        model.to(device)

    # Dataset
    dataset = DCASE2017Task4Dataset()

    # Sampler
    test_sampler = TestSampler(hdf5_path=test_hdf5_path, batch_size=batch_size)

    evaluate_sampler = TestSampler(hdf5_path=evaluate_hdf5_path, batch_size=batch_size)

    # Data loader
    test_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    evaluate_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=evaluate_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    # Evaluator
    evaluator = Evaluator(model=model)

    for (data_type, data_loader, reference_csv_path) in [
        ('test', test_loader, test_reference_csv_path), 
        ('evaluate', evaluate_loader, evaluate_reference_csv_path)]:

        print('Inferencing {} data in about 1 min ...'.format(data_type))

        (statistics, output_dict) = evaluator.evaluate(
            data_loader, reference_csv_path, tmp_submission_path)

        prediction_path = os.path.join(predictions_dir, 
            '{}_iterations.prediction.{}.pkl'.format(iteration, data_type))

        # write_out_prediction(output_dict, prediction_path)
        pickle.dump(output_dict, open(prediction_path, 'wb'))
        print('Write out to {}'.format(prediction_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int)
    parser_train.add_argument('--stop_iteration', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)

    # Inference
    parser_inference_prob = subparsers.add_parser('inference_prob')
    parser_inference_prob.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_inference_prob.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_inference_prob.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_inference_prob.add_argument('--model_type', type=str, required=True)
    parser_inference_prob.add_argument('--loss_type', type=str, required=True)
    parser_inference_prob.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_inference_prob.add_argument('--batch_size', type=int, required=True)
    parser_inference_prob.add_argument('--iteration', type=int, required=True)
    parser_inference_prob.add_argument('--cuda', action='store_true', default=False)
    
    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_prob':
        inference_prob(args)

    else:
        raise Exception('Error argument!')