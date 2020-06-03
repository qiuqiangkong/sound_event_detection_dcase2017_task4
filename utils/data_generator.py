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
    def __init__(self):
        """DCASE 2017 Task 4 dataset."""
        pass
 
    def __getitem__(self, meta):
        """Get input and target data of an audio clip.

        Args:
          meta: dict, e.g., {'hdf5_path':, xxx.h5, 'index_in_hdf5': 34}

        Returns:
          data_dict: {'audio_name': str, 
                      'waveform': (audio_samples,), 
                      'target': (classes_num,), 
                      (ifexist) 'strong_target': (frames_num, classes_num)}
        """

        hdf5_path = meta['hdf5_path']
        index_in_hdf5 = meta['index_in_hdf5']
        data_dict = {}

        with h5py.File(hdf5_path, 'r') as hf:
            audio_name = hf['audio_name'][index_in_hdf5].decode()
            waveform = int16_to_float32(hf['waveform'][index_in_hdf5])
            target = hf['target'][index_in_hdf5].astype(np.float32)

            data_dict = {
                'audio_name': audio_name, 'waveform': waveform, 'target': target}

            if 'strong_target' in hf.keys():
                strong_target = hf['strong_target'][index_in_hdf5].astype(np.float32)
                data_dict['strong_target'] = strong_target

        return data_dict


class TrainSampler(object):
    def __init__(self, hdf5_path, batch_size, random_seed=1234):
        """Training data sampler.
        
        Args:
          hdf5_path, str
          batch_size: int
          random_seed: int
        """
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        with h5py.File(self.hdf5_path, 'r') as hf:
            self.audios_num = len(hf['audio_name'])

        logging.info('Training audio num: {}'.format(self.audios_num))
        self.audio_indexes = np.arange(self.audios_num)        
        self.random_state.shuffle(self.audio_indexes)

        self.pointer = 0

    def __iter__(self):
        """Generate batch meta.
        
        Returns: 
          batch_meta: [{'hdf5_path':, xxx.h5, 'index_in_hdf5': 34},
                       {'hdf5_path':, xxx.h5, 'index_in_hdf5': 12},
                       ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                index = self.audio_indexes[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.audio_indexes)
                
                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': self.audio_indexes[index]})
                i += 1

            yield batch_meta


class TestSampler(object):
    def __init__(self, hdf5_path, batch_size):
        """Testing data sampler.
        
        Args:
          hdf5_path, str
          batch_size: int
        """
        super(TestSampler, self).__init__()
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size

        with h5py.File(self.hdf5_path, 'r') as hf:
            self.audios_num = len(hf['audio_name'])
            
        logging.info('Test audio num: {}'.format(self.audios_num))
        self.audio_indexes = np.arange(self.audios_num)        

    def __iter__(self):
        """Generate batch meta for test. 
        
        Returns: 
          batch_meta: [{'hdf5_path':, xxx.h5, 'index_in_hdf5': 34},
                       {'hdf5_path':, xxx.h5, 'index_in_hdf5': 12},
                       ...]
        """
        batch_size = self.batch_size
        pointer = 0

        while pointer < self.audios_num:
            batch_indexes = np.arange(pointer, 
                min(pointer + batch_size, self.audios_num))

            batch_meta = []

            for index in batch_indexes:
                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': self.audio_indexes[index]})

            pointer += batch_size
            yield batch_meta


def collate_fn(list_data_dict):
    """Collate data.

    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (audio_samples,), ...}, 
                             {'audio_name': str, 'waveform': (audio_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, audio_samples), ...}
    """
    np_data_dict = {}
    
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict
