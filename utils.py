import pandas as pd
import glob
import re

def get_audios(type=''):
    """
    Get all audio files available for a given type (given/custom, i.e. self recorded)

    Parameters:
        - type [str] : 'given'/'custom' to select only one type of files
    Returns:
        - files [list[str]] : names of audio files
        - labels [list[str]] : labels of those files ('adroite', 'agauche'...)
    """
    pattern = './data_*/*'
    if type == 'given':
        pattern = './data_given/*'
    if type == 'custom':
        pattern = './data_custom/*'
    if type == 'veg':
        pattern = './data_veg/*'
    if type == 'veg2':
        pattern = './data_veg2/*'

    files = glob.glob(pattern)
    labels = [re.findall('[a-z]+', f.split('.')[1])[-1] for f in files]

    return files, labels