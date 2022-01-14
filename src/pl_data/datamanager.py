import webknossos as wk
from io import BytesIO
import numpy as np
from tensorflow.python.lib.io import file_io


def read_gcs(file):
    f_stream = file_io.FileIO(file, 'rb')
    return np.load(BytesIO(f_stream.read()))


class GetData():
    """Class for managing different dataset types."""
    def __init__(self, path, token=None):
        self.path = path

        # Interpret the dataset type:
        if "gs://" in path:
            # We have google data, use read_gcs
            data_type = "GCS"
        elif "webknossos" in path:
            # We have webknossos data, use wk
            data_type = "WK"
        else:
            raise NotImplementedError(
                "Could not recognize the data resource: {}".format(
                    path))
        self.data_type = data_type
        self.token = token

    def load(self):
        if self.data_type == "GCS":
            ds = read_gcs(self.path)
            # Check that ds is an npz
            assert ".npz" in self.path, "GCS is only set up for .npz files."
            volume = ds["volume"]
            label = ds["label"]
            del ds.f
            ds.close()
            return volume, label
        elif self.data_type == "WK":

            # 1. Have token in the config
            # 2. Specify a processing strategy
            with wk.webknossos_context(
                    url="https://webknossos.org",
                    token=self.token):
                # First get the annotations
                annotation = wk.Annotation.download(self.path)
                import pdb;pdb.set_trace()

                # Then get the dataset
                ds_name = path.split("/")[-2]
                train_dataset = wk.download_dataset(
                    "zebrafish_vertebra_250um",  # zebrafish_vertebra_250um",
                    "b2275d664e4c2a96",
                    bbox=BoundingBox((10533, 7817, 3547), (1152, 1152, 384)),
                    layers=["color"],  # , "Volume Layer"],
                    mags=[Mag("1")],
                    path="../zebrafish",
                )
