'''
a set of useful tools commonly used by me
'''

import os
import glob
import logging
from datetime import datetime


def maybe_mkdir(directory, new_folder_name):
    try:
        os.makedirs(os.path.join(directory, new_folder_name), exist_ok=True)
        return os.path.join(directory, new_folder_name)
    except OSError:
        logging.debug('directory cannot be created at: ', new_folder_name)


def setup_logging(level=logging.INFO,
                  log_path=str(os.path.join(maybe_mkdir(os.getcwd(), 'logs'), str(datetime.now().date()) + '.txt'))):
    log = logging.getLogger(__name__)
    logging.basicConfig(level=level)
    handler = logging.FileHandler(filename=log_path)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")  # format string for log messages
    handler.setFormatter(formatter)  # handler included for use with pyradiomics
    return log

