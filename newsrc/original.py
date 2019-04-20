
import sys

import numpy as np
import tensorflow as tf

import utility

sys.path.append("src")

from sample import sample_sequence

def model_or_sample(batch_size):

    return sample_sequence(hparams=utility.get_hparams(),
                            length=utility.SAMPLE_LENGTH,
                            start_token=utility.get_start_token()[0],
                            batch_size=batch_size)
