import webknossos as wk
from io import BytesIO
import numpy as np
from tensorflow.python.lib.io import file_io
from time import gmtime, strftime


def read_gcs(file):
    f_stream = file_io.FileIO(file, 'rb')
    return np.load(BytesIO(f_stream.read()))


class GetData():
    """Class for managing different dataset types."""
    def __init__(self, path, cfg):
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
        self.token = cfg.token
        self.scale = cfg.scale
        self.annotation_type = cfg.annotation_type

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
            assert self.token is not None, "You need to pass a token for WK."
            assert self.scale is not None, "You must specify dataset scale."
            assert self.annotation_type in ["nml", "volumetric"], "You must specify annotation_type {'nml', 'volumetric'}"  # noqa
            with wk.webknossos_context(
                    url="https://webknossos.org",
                    token=self.token):
                annotation = wk.Annotation.download(self.path)
                import pdb;pdb.set_trace()
                time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
                new_dataset_name = annotation.dataset_name + f"_segmented_{time_str}"  # noqa
                annotation_layer = annotation.save_volume_annotation(dataset)
                bbox = annotation_layer.bounding_box

                # Either extract nml data or volumetric data
                if self.annotation_type == "nml":
                    # These are synapse annotations
                    import pdb;pdb.set_trace()

                elif self.annotation_type == "volumetric":
                    # These annotations are volumetric, for semantic seg.
                    raise NotImplementedError("Need to finish this")
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
