import webknossos as wk
from io import BytesIO
import numpy as np
from tensorflow.python.lib.io import file_io
from time import gmtime, strftime
from skimage.transform import resize
from webknossos.geometry import Mag, BoundingBox
import fastremap


def draw_cube(start, label_size, shape, label, dtype):
    """Draw a cube in a volume."""
    vol = np.zeros(shape, dtype=dtype)
    vol[
        start[0]: start[0] + label_size[0],
        start[1]: start[1] + label_size[1],
        start[2]: start[2] + label_size[2],
    ] = label
    return vol


def read_gcs(file):
    f_stream = file_io.FileIO(file, 'rb')
    return np.load(BytesIO(f_stream.read()))


class GetData():
    """Class for managing different dataset types."""
    def __init__(self, cfg):
        self.image_path = cfg.image_path
        self.annotation_path = cfg.annotation_path

        # Interpret the dataset type:
        if "gs://" in self.annotation_path:
            # We have google data, use read_gcs
            data_type = "GCS"
        elif "webknossos" in self.annotation_path:
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
        self.label_transpose_xyz_zyx = cfg.label_transpose_xyz_zyx
        self.image_downsample = cfg.image_downsample
        self.label_downsample = cfg.label_downsample
        self.image_layer_name = cfg.image_layer_name
        self.cube_size = cfg.cube_size

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
            assert self.image_layer_name is not None, "You must specify an image layer name. (images? color?)"  # noqa
            assert self.cube_size is not None, "You must specify an image cube size"  # noqa
            with wk.webknossos_context(
                    url="https://webknossos.org",
                    token=self.token):
                annotation = wk.Annotation.download(self.annotation_path)
                original_dataset_name = self.wkdataset.split("/")[-1]
                original_dataset_org = self.wkdataset.split("/")[-2]
                time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
                new_dataset_name = annotation.dataset_name + f"_segmented_{time_str}"  # noqa

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

                    min_coords = np.maximum(
                        coords.min(0) - self.annotation_size,
                        np.asarray([0, 0, 0]))
                    max_coords = coords.max(0) + self.annotation_size
                    diffs = max_coords - min_coords
                    min_coords = min_coords.astype(int)
                    diffs = diffs.astype(int)
                    bbox = BoundingBox(min_coords.tolist(), diffs.tolist())

                    # The below WK loading works, but is missing
                    # the extra membrane prediction layer.
                    # Not using for now
                    #
                    # # Then get the dataset images
                    # dataset = wk.download_dataset(
                    #     original_dataset_name,
                    #     original_dataset_org,
                    #     bbox=bbox,
                    #     layers=[self.image_layer_name],  # , "Volume Layer"],
                    #     mags=[Mag("1")],
                    #     path="../{}".format(new_dataset_name),
                    # )
                    # image_layer = dataset.get_layer(self.image_layer_name)
                    # image_mag = image_layer.get_mag(Mag("1"))
                    # volume = image_mag.read().squeeze(0)
                    volume = read_gcs(self.image_path)

                    # Create annotation image
                    annotation_size = np.asarray(self.annotation_size).astype(int)  # noqa
                    cube_size = np.asarray(self.cube_size).astype(int)
                    label_shape = np.ceil(np.asarray(volume.shape[:-1]) * self.label_downsample).astype(int)[::-1]  # noqa
                    dtype = volume.dtype
                    label_vol = np.zeros((label_shape), dtype=dtype)
                    for label, coord in zip(labels, coords):
                        startc = np.maximum(coord - (cube_size // 2), np.zeros_like(coord))  # noqa
                        startc = startc.astype(int)
                        endc = np.minimum(startc + cube_size, label_shape)
                        label_cube_size = endc - startc
                        cube = draw_cube(
                            cube_size // 2 - annotation_size // 2,  # top-left edge of label  # noqa
                            annotation_size,  # label size
                            label_cube_size,  # volume shape
                            dtype=dtype,
                            label=label)

                        label_vol[
                            startc[0]: endc[0],
                            startc[1]: endc[1],
                            startc[2]: endc[2]] = cube

                    # Transpose images if requested
                    if self.image_transpose_xyz_zyx:
                        volume = volume.transpose(
                            self.image_transpose_xyz_zyx)

                    if self.label_transpose_xyz_zyx:
                        label_vol = label_vol.transpose(
                            self.label_transpose_xyz_zyx)

                    # Downsample images if requested.
                    import pdb;pdb.set_trace()
                    if self.label_downsample:
                        label_vol = resize(
                            label_vol,
                            image_downsample,
                            anti_aliasing=True,
                            preserve_range=True,
                            order=1).astype(dtype)
                    if self.image_downsample:
                        # Upsample volume to native res
                        volume = resize(
                            volume,
                            label_vol.shape[:-1],
                            anti_aliasing=True,
                            preserve_range=True,
                            order=3).astype(dtype)

                    return volume, label_vol

                elif self.annotation_type == "volumetric":
                    # These annotations are volumetric, for semantic seg.
                    dataset = wk.Dataset(new_dataset_name, scale=list(self.scale))
                    annotation_layer = annotation.save_volume_annotation(dataset)
                    bbox = annotation_layer.bounding_box
                    raise NotImplementedError("Need to finish this")
                    return volume, label_vol
