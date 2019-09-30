import numpy as np 
import logging
import os
from utils.log_func import get_log_func
import sys
import json

log_level = os.getenv("LOG_LEVEL", "WARNING")
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
log = get_log_func(__name__)

def mean_masked_entropy(probs, y_true, pad_id):
    """ calculate the mean entropy of a (masked) probability distribution -> probs * log(probs)"""

    mask = np.not_equal(y_true, pad_id)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
    entropy *= mask  # NOTE mask is 0 at positions to ignore
    # average over unmasked sequence positions, then over samples
    return np.mean(np.sum(entropy, axis=-1) / np.sum(mask, axis=-1))

def on_train_start(config):
    """ sets up checkpoint folders and logging according to the config

        Args:
            config: config file
     
    """

    # save all checkpoint folders to checkpoint dir
    working_dir = os.path.join("checkpoints", config['data']['working_dir'])
    vocab_dir = os.path.join("checkpoints", config['data']['vocab_dir'])
    lm_dir = os.path.join("checkpoints", config['data']['lm_dir'])

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
    if not os.path.exists(lm_dir):
        os.makedirs(lm_dir)

    config_path = os.path.join(working_dir, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
    return working_dir, vocab_dir, lm_dir    

class EarlyStopping(object):
    """ class that monitors a metric and stops training when the metric has stopped improving
    
        Args: 
            monitor_improvement - if True, stops training when metric stops increasing. If False, 
                when metric stops decreasing. 
            patience - after how many epochs of degrading performance should training be stopped
    """

    def __init__(self, increase_good = True, patience = 0):

        self.increase_good = increase_good
        self.patience = patience
        self.prev_metric = None
        self.degrade_count = 0

    def __call__(self, cur_metric):
        
        if self.prev_metric is None:
            self.prev_metric = cur_metric
            return
        else:
            if self.increase_good:
                if cur_metric < self.prev_metric:
                    self.degrade_count += 1
                else:
                    self.degrade_count = 0
            else:
                if cur_metric > self.prev_metric:
                    self.degrade_count += 1
                else:
                    self.degrade_count = 0
            if self.degrade_count > patience:
                sys.exit(f'Metric has degraded for {self.degrade_count} epochs, exiting training')
            self.prev_metric = cur_metric

