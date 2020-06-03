import numpy as np
import logging
from sklearn import metrics

from pytorch_utils import forward
from utilities import (get_filename, frame_prediction_to_event_prediction, 
    write_submission, official_evaluate)
import config


def sed_average_precision(strong_target, framewise_output, average):
    """Calculate framewise SED mAP.

    Args:
      strong_target: (N, frames_num, classes_num)
      framewise_output: (N, frames_num, classes_num)
      average: None | 'macro' | 'micro'
    """
    assert strong_target.shape == framewise_output.shape
    (N, time_steps, classes_num) = strong_target.shape

    average_precision = metrics.average_precision_score(
        strong_target.reshape((N * time_steps, classes_num)), 
        framewise_output.reshape((N * time_steps, classes_num)), 
        average=average)
    
    return average_precision


class Evaluator(object):
    def __init__(self, model):
        """Evaluator.

        Args:
          model: object, model to be used for inference
        """
        self.model = model
        
        self.labels = config.labels
        self.idx_to_lb = config.idx_to_lb
    
        # Default parameters for SED
        self.sed_params_dict = {
            'audio_tagging_threshold': 0.5, 
            'sed_high_threshold': 0.5, 
            'sed_low_threshold': 0.2, 
            'n_smooth': 10, 
            'n_salt': 10}

    def evaluate(self, data_loader, reference_csv_path, submission_path):
        """Evaluate AT and SED performance.

        Args:
          data_loader: object
          reference_csv_path: str, strongly labelled ground truth csv
          submission: str, path to write out submission file

        Returns:
          statistics: dict
          output_dict: dict
        """
        output_dict = forward(
            model=self.model, 
            data_loader=data_loader, 
            return_input=False, 
            return_target=True)
        
        statistics = {}

        # Clipwise statistics
        statistics['clipwise_ap'] = metrics.average_precision_score(
            output_dict['target'], output_dict['clipwise_output'], average=None)

        # Framewise statistics
        if 'strong_target' in output_dict.keys():
            statistics['framewise_ap'] = sed_average_precision(
                output_dict['strong_target'], 
                output_dict['framewise_output'], average=None)
         
        # Framewise predictions to eventwise predictions
        predict_event_list = frame_prediction_to_event_prediction(output_dict, 
            self.sed_params_dict)
        
        # Write eventwise predictions to submission file
        write_submission(predict_event_list, submission_path)

        # SED with official tool
        statistics['sed_metrics'] = official_evaluate(reference_csv_path, submission_path)

        return statistics, output_dict