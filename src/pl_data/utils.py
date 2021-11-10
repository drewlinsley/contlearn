from io import StringIO
import tensorflow as tf
import numpy as np
from tensorflow.python.lib.io import file_io


def read_gcs(file):
    import pdb;pdb.set_trace()
    data = StringIO(file_io.read_file_to_string(file))
