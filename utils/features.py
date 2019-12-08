import os
import sys
import numpy as np
import argparse
import h5py
import librosa
import matplotlib.pyplot as plt
import time
import csv
import math
import re
import random

import config
from utilities import create_folder, pad_truncate_sequence, float32_to_int16


def read_weak_csv(weak_label_csv_path, data_type):
    """Read weakly labelled ground truth csv file. 

    Args:
      weak_label_csv_path: str
      data_type: 'training' | 'testing' | 'evaluation'

    Returns:
      meta_list: [{'audio_name': 'a.wav', 'labels': ['Train', 'Bus']}
                  ...]
    """
    assert data_type in ['training', 'testing', 'evaluation']
    
    if data_type in ['training', 'testing']:
        with open(weak_label_csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            rows = list(reader)
            
    elif data_type in ['evaluation']:
        with open(weak_label_csv_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            rows = list(reader)
            
    meta_list = []
            
    for row in rows:
        if data_type in ['training', 'testing']:
            meta = {
                'audio_name': 'Y' + row[0] + '_' + row[1] + '_' + row[2] + '.wav', 
                'labels': re.split(',(?! )', row[3])}
            meta_list.append(meta)
        
        elif data_type in ['evaluation']:
            audio_name = row[0]
            name_list = [meta['audio_name'][1 :] for meta in meta_list]

            if audio_name in name_list:
                n = name_list.index(audio_name)
                meta_list[n]['labels'].append(row[3])
            else:
                meta = {
                    'audio_name': 'Y{}'.format(row[0]), 
                    'labels': [row[3]]}
                meta_list.append(meta)

    return meta_list


def get_weak_csv_filename(data_type):
    """Prepare weakly labelled csv path. 

    Args:
      data_type: 'training' | 'testing' | 'evaluation'

    Returns:
      str, weakly labelled csv path
    """
    if data_type in ['training', 'testing']:
        return '{}_set.csv'.format(data_type)
        
    elif data_type in ['evaluation']:
        return 'groundtruth_weak_label_evaluation_set.csv'
        
    else:
        raise Exception('Incorrect argument!')


def read_strong_csv(strong_meta_csv_path):
    """Read strongly labelled ground truth csv file. 
    
    Args:
      strong_meta_csv_path: str

    Returns: 
      meta_dict: {'a.wav': [{'begin_time': 3.0, 'end_time': 5.0, 'label': 'Bus'}
                            {'begin_time': 4.0, 'end_time': 7.0, 'label': 'Train'}
                            ...]
                  ...
                 }
    """
    with open(strong_meta_csv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter='\t')
        lines = list(reader)
        
    meta_dict = {}
    for line in lines:
        [audio_name, begin_time, end_time, label] = line
        meta = {'begin_time': begin_time, 'end_time': end_time, 'label': label}
        if audio_name in meta_dict:
            meta_dict[audio_name].append(meta)
        else:
            meta_dict[audio_name] = [meta]
        
    return meta_dict


def get_weak_target(labels, lb_to_idx):
    """Reformat weakly labelled target to vector format. 

    Args:
      labels: list of str
      lb_to_idx: dict

    Returns:
      target: (classes_num,)
    """
    classes_num = len(lb_to_idx)
    target = np.zeros(classes_num, dtype=np.bool)
    
    for label in labels: 
        target[lb_to_idx[label]] = True
        
    return target 


def get_strong_target(audio_name, strong_meta_dict, frames_num, 
    frames_per_second, lb_to_idx):
    """Reformat strongly labelled target to matrix format. 

    Args:
      audio_name: str
      strong_meta_dict: dict
      frames_num: int
      frames_per_second: int
      lb_to_idx: dict

    Returns:
      target: (frames_num, classes_num)
    """
    
    meta_list = strong_meta_dict[audio_name]
    
    target = np.zeros((frames_num, len(lb_to_idx)), dtype=np.bool)
    
    for meta in meta_list:
        begin_time = float(meta['begin_time']) 
        begin_frame = int(round(begin_time * frames_per_second))
        end_time = float(meta['end_time'])
        end_frame = int(round(end_time * frames_per_second)) + 1
        label = meta['label']
        idx = lb_to_idx[label]
        
        target[begin_frame : end_frame, idx] = 1
    
    return target


def pack_audio_files_to_hdf5(args):
    """Pack waveform to hdf5 file. 

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, Directory of your workspace
      data_type: 'training' | 'testing' | 'evaluation'
      mini_data: bool, set True for debugging on a small part of data
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    data_type = args.data_type
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    audio_length = config.audio_length
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx
    frames_per_second = config.frames_per_second
    frames_num = frames_per_second * config.audio_duration

    has_strong_target = data_type in ['testing', 'evaluation']

    # Paths
    audios_dir = os.path.join(dataset_dir, data_type)
    weak_label_csv_path = os.path.join(dataset_dir, 'metadata', 
        get_weak_csv_filename(data_type))

    if data_type == 'testing':
        strong_label_csv_path = os.path.join(dataset_dir, 'metadata', 
            'groundtruth_strong_label_testing_set.csv')
    elif data_type == 'evaluation':
        strong_label_csv_path = os.path.join(dataset_dir, 'metadata', 
            'groundtruth_strong_label_evaluation_set.csv')

    if mini_data:
        packed_hdf5_path = os.path.join(workspace, 'features', 
            'minidata_{}.waveform.h5'.format(data_type))
    else:
        packed_hdf5_path = os.path.join(workspace, 'features', 
            '{}.waveform.h5'.format(data_type))
    create_folder(os.path.dirname(packed_hdf5_path))

    # Read metadata
    weak_meta_list = read_weak_csv(weak_label_csv_path, data_type)

    # Use a small amount of data for debugging
    if mini_data:
        random.seed(1234)
        random.shuffle(weak_meta_list)
        weak_meta_list = weak_meta_list[0 : 100]

    audios_num = len(weak_meta_list)

    feature_time = time.time()
    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='audio_name', 
            shape=(audios_num,), 
            dtype='S80')

        hf.create_dataset(
            name='waveform', 
            shape=(audios_num, audio_length), 
            dtype=np.int32)

        hf.create_dataset(
            name='weak_target', 
            shape=(audios_num, classes_num), 
            dtype=np.float32)

        if has_strong_target:
            strong_meta_dict = read_strong_csv(strong_label_csv_path)        
            
            hf.create_dataset(
                name='strong_target', 
                shape=(0, frames_num, classes_num), 
                maxshape=(None, frames_num, classes_num), 
                dtype=np.bool)

        for n in range(audios_num):
            print(n)
            weak_meta_dict = weak_meta_list[n]
            audio_name = weak_meta_dict['audio_name']
            audio_path = os.path.join(audios_dir, audio_name)
            (audio, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
            audio = pad_truncate_sequence(audio, audio_length)

            hf['audio_name'][n] = audio_name.encode()
            hf['waveform'][n] = float32_to_int16(audio)
            hf['weak_target'][n] = weak_target = get_weak_target(
                weak_meta_dict['labels'], lb_to_idx)

            if has_strong_target:
                strong_target = get_strong_target(
                    weak_meta_dict['audio_name'][1:], strong_meta_dict, 
                    frames_num, frames_per_second, lb_to_idx)
                
                hf['strong_target'].resize((n + 1, frames_num, classes_num))
                hf['strong_target'][n] = strong_target

    print('Write hdf5 to {}'.format(packed_hdf5_path))
    print('Time: {:.3f} s'.format(time.time() - feature_time))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Pack waveform to hdf5 file
    parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5')
    parser_pack_audio.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_audio.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_pack_audio.add_argument('--data_type', type=str, choices=['training', 'testing', 'evaluation'], required=True, help='Directory of your workspace.')
    parser_pack_audio.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'pack_audio_files_to_hdf5':
        pack_audio_files_to_hdf5(args)
        
    else:
        raise Exception('Incorrect arguments!')