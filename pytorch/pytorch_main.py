import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
from evaluate import Evaluator
from config import (sample_rate, classes_num, mel_bins, fmin, fmax, 
    window_size, hop_size, window, pad_mode, center, device, ref, amin, top_db)
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup
from utilities import (create_folder, get_filename, create_logging, 
    write_out_prediction, StatisticsContainer, Mixup)
from data_generator import (DCASE2017Task4Dataset, TrainSampler, 
    EvaluateSampler, Collator)
from models import *


def train(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    few_shots = args.few_shots
    random_seed = args.random_seed
    resume_iteration = args.resume_iteration
    stop_iteration = args.stop_iteration
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    mini_data = args.mini_data
    filename = args.filename

    loss_func = get_loss_func(loss_type)
    pretrain = True if pretrained_checkpoint_path else False
    num_workers = 16
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    train_hdf5_path = os.path.join(workspace, 'features', 
        '{}training.waveform.h5'.format(prefix))

    test_hdf5_path = os.path.join(workspace, 'features', 
        'testing.waveform.h5'.format(prefix))

    evaluate_hdf5_path = os.path.join(workspace, 'features', 
        'evaluation.waveform.h5'.format(prefix))

    test_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_testing_set.csv')
        
    evaluate_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_evaluation_set.csv')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'holdout_fold={}'.format(holdout_fold), model_type, 
        'pretrain={}'.format(pretrain), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'few_shots={}'.format(few_shots), 'random_seed={}'.format(random_seed), 
        'freeze_base={}'.format(freeze_base))
    create_folder(checkpoints_dir)

    tmp_submission_path = os.path.join(workspace, '_tmp_submission', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'pretrain={}'.format(pretrain), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'few_shots={}'.format(few_shots), 'random_seed={}'.format(random_seed), 
        'freeze_base={}'.format(freeze_base), '_submission.csv')
    create_folder(os.path.dirname(tmp_submission_path))

    statistics_path = os.path.join(workspace, 'statistics', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'pretrain={}'.format(pretrain), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'few_shots={}'.format(few_shots), 'random_seed={}'.format(random_seed), 
        'freeze_base={}'.format(freeze_base), 'statistics.pickle')
    create_folder(os.path.dirname(statistics_path))

    predictions_dir = os.path.join(workspace, 'predictions', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'pretrain={}'.format(pretrain), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
        'few_shots={}'.format(few_shots), 'random_seed={}'.format(random_seed), 
        'freeze_base={}'.format(freeze_base), 'batch_size={}'.format(batch_size))
    create_folder(predictions_dir)

    logs_dir = os.path.join(workspace, 'logs', '{}{}'.format(prefix, filename), 
        'holdout_fold={}'.format(holdout_fold), model_type, 
        'pretrain={}'.format(pretrain), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'few_shots={}'.format(few_shots), 
        'random_seed={}'.format(random_seed), 'freeze_base={}'.format(freeze_base), 
        'batch_size={}'.format(batch_size))
    create_logging(logs_dir, 'w')
    logging.info(args)

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
    
    # Model
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
        classes_num)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)

    if resume_iteration:
        resume_checkpoint_path = os.path.join(checkpoints_dir, '{}_iterations.pth'.format(resume_iteration))
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

    train_dataset = DCASE2017Task4Dataset(hdf5_path=train_hdf5_path)
    test_dataset = DCASE2017Task4Dataset(hdf5_path=test_hdf5_path)
    evaluate_dataset = DCASE2017Task4Dataset(hdf5_path=evaluate_hdf5_path)

    train_sampler = TrainSampler(
        hdf5_path=train_hdf5_path, 
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size, 
        few_shots=few_shots, 
        random_seed=random_seed)

    test_sampler = EvaluateSampler(dataset_size=len(test_dataset), batch_size=batch_size)
    evaluate_sampler = EvaluateSampler(dataset_size=len(evaluate_dataset), batch_size=batch_size)

    collector = Collator()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_sampler=train_sampler, collate_fn=collector, 
        num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
        batch_sampler=test_sampler, collate_fn=collector, 
        num_workers=num_workers, pin_memory=True)

    evaluate_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
        batch_sampler=evaluate_sampler, collate_fn=collector, 
        num_workers=num_workers, pin_memory=True)

    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)
        
    # Evaluator
    test_evaluator = Evaluator(
        model=model, 
        generator=test_loader)

    evaluate_evaluator = Evaluator(
        model=model, 
        generator=evaluate_loader)

    train_bgn_time = time.time()
    
    # Train on mini batches
    for batch_data_dict in train_loader:
        
        # Evaluate
        if iteration % 1000 == 0:
            if resume_iteration > 0 and iteration == resume_iteration:
                pass
            else:
                logging.info('------------------------------------')
                logging.info('Iteration: {}'.format(iteration))

                train_fin_time = time.time()

                for (data_type, evaluator, reference_csv_path) in [
                    ('test', test_evaluator, test_reference_csv_path), 
                    ('evaluate', evaluate_evaluator, evaluate_reference_csv_path)]:

                    logging.info('{} statistics:'.format(data_type))

                    (statistics, predictions) = evaluator.evaluate(
                        reference_csv_path, tmp_submission_path)

                    statistics_container.append(data_type, iteration, statistics)

                    prediction_path = os.path.join(predictions_dir, 
                        '{}_iterations.prediction.{}.h5'.format(iteration, data_type))

                    write_out_prediction(predictions, prediction_path)
                
                statistics_container.dump()

                train_time = train_fin_time - train_bgn_time
                validate_time = time.time() - train_fin_time

                logging.info(
                    'Train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(train_time, validate_time))

                train_bgn_time = time.time()

        # Save model 
        if iteration % 10000 == 0 and iteration > 49999:
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                'optimizer': optimizer.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        
        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(len(batch_data_dict['waveform']))

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
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--few_shots', type=int, required=True)
    parser_train.add_argument('--random_seed', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int)
    parser_train.add_argument('--stop_iteration', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')