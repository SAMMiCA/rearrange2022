# From MaSS 
# https://github.com/brandontrabucco/mass/blob/main/mass/utils/experimentation.py
import numpy as np
import torch
import json
import os
import stat
import signal

from typing import Callable, Dict, List, Optional, Tuple, Union, Set
from scipy.optimize import linear_sum_assignment

from rearrange.tasks import UnshuffleTask
from ai2thor.exceptions import RestartError, UnityCrashException


class NumpyJSONEncoder(json.JSONEncoder):
    
    """JSON encoder for numpy objects.

    Based off the stackoverflow answer by Jie Yang here: https://stackoverflow.com/a/57915246.
    The license for this code is [BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
    """
    
    def default(self, obj):
        if isinstance(obj, np.void):
            return None
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)
        
        
class TimeoutDueToUnityCrash(object):
    """Helper class for detecting when the unity server cannot be reached,
    which indicates that unity has likely crashed due to a cause that
    AI2-THOR does not handle, which causes infinite blocking.

    """
    
    def __init__(self, seconds: int = 60):
        self.seconds = seconds  # time to wait before raising an error
        
    def handle_timeout(self, signum, frame):
        raise UnityCrashException(
            f"Unity server did not respond in {self.seconds} seconds."
        )
    
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)  # wait before throwing an error
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # call finished, disable the alarm
        
        
def run_experiment_with_restart(run_experiment: Callable, *args, **kwargs):
    """Helper function for running experiments in AI2-THOR that handles when
    unity crashes and restarts the experiment so that restarts do not
    interfere with experiments running to completion.

    """
    
    while True:
        try:
            return run_experiment(*args, **kwargs)
        except (RestartError, UnityCrashException) as e:
            print(f"Restarting experiment due to: {e}")
        except Exception as e:
            raise e


def handle_read_only(func, path, exc_info):
    """Helper function that allows shutil to recursively delete the temporary
    folder generated in this experiment, which contains files with read-only
    file access, which causes an error and must be modified.

    """
    
    # check if the path has read only access currently
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        func(path)