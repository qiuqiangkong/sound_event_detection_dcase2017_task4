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
    def __init__(self, model, generator):
        """Evaluator.

        Args:
          model: object, model to be used for inference
          generator: object, generator of data to be evaluated
        """
        self.model = model
        self.generator = generator
        
        self.labels = config.labels
        self.idx_to_lb = config.idx_to_lb
    
        # Default parameters for SED
        self.sed_params_dict = {
            'audio_tagging_threshold': 0.5, 
            'sed_high_threshold': 0.5, 
            'sed_low_threshold': 0.2, 
            'n_smooth': 10, 
            'n_salt': 10}

    def evaluate(self, reference_csv_path, submission_path):
        """Evaluate AT and SED performance.

        Args:
          reference_csv_path: str, strongly labelled ground truth csv
          submission: str, path to write out submission file
        """
        output_dict = forward(
            model=self.model, 
            generator=self.generator, 
            return_input=False, 
            return_target=True)

        predictions = {'clipwise_output': output_dict['clipwise_output'], 
            'framewise_output': output_dict['framewise_output']}

        statistics = {}
        
        # Weak statistics
        clipwise_ap = metrics.average_precision_score(
            output_dict['target'], output_dict['clipwise_output'], average=None)
        statistics['clipwise_ap'] = clipwise_ap
        logging.info('    clipwise mAP: {:.3f}'.format(np.mean(clipwise_ap)))

        if 'strong_target' in output_dict.keys():
            framewise_ap = sed_average_precision(output_dict['strong_target'], 
                output_dict['framewise_output'], average=None)
            statistics['framewise_ap'] = framewise_ap
            logging.info('    framewise mAP: {:.3f}'.format(np.mean(framewise_ap)))
         
        # Obtain eventwise prediction frame framewise prediction using predefined thresholds
        predict_event_list = frame_prediction_to_event_prediction(output_dict, 
            self.sed_params_dict)
        
        # Write predicted events to submission file
        write_submission(predict_event_list, submission_path)

        # SED with official tool
        results = official_evaluate(reference_csv_path, submission_path)
        logging.info('    {}'.format(results['overall']['error_rate']))
        statistics['sed_metrics'] = results

        return statistics, predictions