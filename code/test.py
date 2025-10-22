"""
Testing script for DDSE
"""

from run import DDSE_run


DDSE_run(
    model_name='ddse', 
    dataset_name='mosi', 
    is_tune=False, 
    seeds=[1111], 
    model_save_dir="./pt",
    res_save_dir="./result", 
    log_dir="./log", 
    mode='test', 
    is_training=False
)