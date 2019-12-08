import numpy as np
import h5py
import csv
import time
import logging
import os
import glob
import matplotlib.pyplot as plt
import logging

from utilities import int16_to_float32
import config


class DCASE2017Task4Dataset(object):
    def __init__(self, hdf5_path):

        self.hdf5_path = hdf5_path

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
       
        self.audios_num = len(audio_names)
        logging.info('Audio samples: {}'.format(self.audios_num))
 
    def __getitem__(self, index):

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_name = hf['audio_name'][index].decode()
            waveform = int16_to_float32(hf['waveform'][index])
            target = hf['weak_target'][index].astype(np.float32)
                
        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target}
            
        return data_dict
    
    def __len__(self):
        return self.audios_num


class TrainSampler(object):

    def __init__(self, hdf5_path, batch_size, few_shots, random_seed):
        """Inference sampler. Generate audio indexes for DataLoader. 
        
        Args:
          batch_size: int
        """
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            target = hf['weak_target'][:]

        self.train_audio_indexes = np.arange(len(audio_names))

        if few_shots > 0:
            self.random_state.shuffle(self.train_audio_indexes)
            classes_num = target.shape[-1]
            new_indexes = []
            for k in range(classes_num):
                new_indexes.append(self.train_audio_indexes[np.where(target[self.train_audio_indexes][:, k] == 1)[0][0 : few_shots]])
            self.train_audio_indexes = np.concatenate(new_indexes)

        logging.info('Training audio num: {}'.format(len(self.train_audio_indexes)))
        
        self.random_state.shuffle(self.train_audio_indexes)

    def __iter__(self):
        """Generate audio indexes for evaluation.
        
        Returns: batch_indexes: (batch_size,)
        """
        N = len(self.train_audio_indexes)
        self.pointer = 0

        while True:
            # Reset pointer
            if self.pointer >= N:
                self.pointer = 0
                self.random_state.shuffle(self.train_audio_indexes)

            # Get batch audio_indexes
            batch_audio_indexes = self.train_audio_indexes[
                self.pointer: min(self.pointer + self.batch_size, N)]
                
            self.pointer += self.batch_size
            yield batch_audio_indexes


class EvaluateSampler(object):

    def __init__(self, dataset_size, batch_size):
        """Inference sampler. Generate audio indexes for DataLoader. 
        
        Args:
          batch_size: int
        """
        self.batch_size = batch_size
        self.dataset_size = dataset_size

    def __iter__(self):
        """Generate audio indexes for evaluation.
        
        Returns: batch_indexes: (batch_size,)
        """
        batch_size = self.batch_size

        pointer = 0

        while pointer < self.dataset_size:
            batch_indexes = np.arange(pointer, 
                min(pointer + batch_size, self.dataset_size))

            pointer += batch_size
            yield batch_indexes


class Collator(object):
    def __init__(self):
        """Data collator.
        
        Args:
          mixup: bool
        """
        pass
        
    
    def __call__(self, list_data_dict):
        """Collate data to tensor. Add mixup information to list_data_dict. 
        
        Args:
          list_data_dict: 
            [{'audio_name': 'YtwJdQzi7x7Q.wav', 'waveform': (audio_length,), 'target': (classes_num)}, 
            ...]
        Returns:
          np_data_dict: {
            'audio_name': (audios_num,), 
            'waveform': (audios_num, audio_length), 
            'target': (audios_num, classes_num), 
            (optional) 'mixup_lambda': (audios_num,)}
        """
        np_data_dict = {}
        
        np_data_dict['audio_name'] = np.array([data_dict['audio_name'] for data_dict in list_data_dict])
        np_data_dict['waveform'] = np.array([data_dict['waveform'] for data_dict in list_data_dict])
        np_data_dict['target'] = np.array([data_dict['target'] for data_dict in list_data_dict])

        return np_data_dict