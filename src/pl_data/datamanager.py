import webknossos as wk
from io import BytesIO
import numpy as np
from tensorflow.python.lib.io import file_io
from time import gmtime, strftime
from skimage.transform import resize
from webknossos.geometry import Mag
import fastremap


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
        self.wkdataset = cfg.wkdataset
        self.annotation_size = cfg.annotation_size
        self.image_transpose_xyz_zyx = cfg.image_transpose_xyz_zyx
        self.image_downsample = cfg.image_downsample

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
            assert self.wkdataset is not None, "You must specify the original dataset."  # noqa
            with wk.webknossos_context(
                    url="https://webknossos.org",
                    token=self.token):
                annotation = wk.Annotation.download(self.path)
                original_dataset_name = self.wkdataset.split("/")[-1]
                original_dataset_org = self.wkdataset.split("/")[-2]

                # Either extract nml data or volumetric data
                if self.annotation_type == "nml":
                    # These are synapse annotations
                    assert self.annotation_size is not None, "You are using an nml and must specify annotation size"  # noqa
                    original_dataset = annotation.skeleton
                    nml_meta = annotation._nml[0]
                    nml_list = [x for x in annotation._nml[1:] if len(x)]
                    nml_lens = [len(x) for x in nml_list]
                    annotations = nml_list[np.argmax(nml_lens)]
                    nml_label_key = nml_list[-2]

                    # Get the labels and xyzs in each annotation
                    labels, coords = [], []
                    for node in annotations:
                        if len(node.nodes):
                            # Remove bad nodes
                            labels.append(node.groupId)
                            coords.append(node.nodes[0].position)
                    labels = np.asarray(labels)
                    labels, remapping = fastremap.renumber(
                        labels,
                        in_place=True)
                    coords = np.asarray(coords)

                    import pdb;pdb.set_trace()
                    min_coords = np.maximum(
                        coords.min(0) - self.annotation_size,
                        np.asarray([0, 0, 0]))
                    max_coords = coords.max(0) + self.annotation_size
                    diffs = max_coords - min_coords
                    bbox = (
                        tuple(min_coords.tolist()),
                        tuple(diffs.tolist())
                    )

                    # Then get the dataset images
                    dataset = wk.download_dataset(
                        original_dataset_name,
                        original_dataset_org,
                        bbox=bbox,
                        layers=["color"],  # , "Volume Layer"],
                        mags=[Mag("1")],
                        path="../wkdata",
                    )
                    volume = dataset.read()

                    # Transpose images if requested
                    if self.image_transpose_xyz_zyx:
                        volume = volume.transpose(self.image_transpose_xyz_zyx)

                    # Downsample images if requested.
                    if self.image_downsample:
                        volume = resize(
                            volume,
                            image_downsample,
                            anti_aliasing=True,
                            preserve_range=True,
                            order=3)
                        labels = labels / np.asarray(
                            self.image_downsample)[None]
                        labels = labels.astype(int)

                    # Create annotation image
                    label_vol = np.zeros_like(data)
                    for label in labels:
                        label_vol[label[0], label[1], label[2]] = draw_sphere(
                            label,
                            self.annotation_size)
                    return volume, label_vol

                elif self.annotation_type == "volumetric":
                    # These annotations are volumetric, for semantic seg.
                    time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
                    new_dataset_name = annotation.dataset_name + f"_segmented_{time_str}"  # noqa
                    dataset = wk.Dataset(new_dataset_name, scale=list(self.scale))
                    annotation_layer = annotation.save_volume_annotation(dataset)
                    bbox = annotation_layer.bounding_box
                    raise NotImplementedError("Need to finish this")
                    return volume, label_vol
