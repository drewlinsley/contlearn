from io import BytesIO
import numpy as np
from tensorflow.python.lib.io import file_io


def read_gcs(file):
    f_stream = file_io.FileIO(file, 'rb')
    return np.load(BytesIO(f_stream.read()))
