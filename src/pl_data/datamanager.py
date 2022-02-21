import os
import webknossos as wk
from io import BytesIO
import numpy as np
from tensorflow.python.lib.io import file_io
from time import gmtime, strftime
from skimage.transform import resize
from webknossos.geometry import Mag, BoundingBox
import fastremap
from tqdm import tqdm
from joblib import Parallel, delayed
from src.pl_data.augmentation_functions import randomcrop
from scipy import ndimage


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
        if not self.image_downsample:
            self.image_downsample = [1, 1, 1]
        self.label_downsample = cfg.label_downsample
        self.image_layer_name = cfg.image_layer_name
        self.cube_size = cfg.cube_size
        self.bounding_box = cfg.bounding_box
        self.keep_labels = cfg.keep_labels
        self.source_volume_name = cfg.source_volume_name
        self.create_subvolumes = cfg.create_subvolumes
        # self.volume_size = cfg.volume_size

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
            assert self.source_volume_name is not None, "You must specify a source_volume_name or set this to False"
            with wk.webknossos_context(
                    url="https://webknossos.org",
                    token=self.token,
                    timeout=7200):
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

                    # Now prune coords so that we only keep those in self.bounding_box
                    if self.bounding_box:
                        coord_max = np.asarray(self.bounding_box[0]) + np.asarray(self.bounding_box[1])
                        coord_mask = np.all(coords < coord_max, 1)
                        coords = coords[coord_mask]
                        labels = labels[coord_mask]
                    # max_coords = coords.max(0) + self.annotation_size
                    # diffs = max_coords - min_coords
                    # min_coords = min_coords.astype(int)
                    # diffs = diffs.astype(int)
                    # bbox = BoundingBox(min_coords.tolist(), diffs.tolist())

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
                    label_shape = np.ceil(np.asarray(volume.shape[:-1]) / self.image_downsample).astype(int)  # noqa
                    dtype = volume.dtype
                    # label_vol = np.zeros((label_shape), dtype=dtype)
                    volume_list, label_list = [], []  # noqa Create a list of the processed label cubes
                    if self.label_transpose_xyz_zyx:
                        coords = coords[
                            :,
                            self.label_transpose_xyz_zyx]
                        if self.bounding_box:
                            self.bounding_box = np.asarray(self.bounding_box)
                            self.bounding_box[0] = self.bounding_box[0, self.label_transpose_xyz_zyx]
                            self.bounding_box[1] = self.bounding_box[1, self.label_transpose_xyz_zyx]
                    res_coords = np.ceil(coords * self.image_downsample)  # noqa Resize the coordinates
                    # full_cube_size = (cube_size * 1.5).astype(int)

                    # First build the label volume
                    # min_coords = np.maximum(
                    #     res_coords.min(0) - np.asarray(self.annotation_size),
                    #     np.asarray([0, 0, 0]))
                    label_vol = np.zeros_like(volume)[..., 0]
                    filtered_coords = []
                    if np.any(self.bounding_box):
                        min_coords = self.bounding_box[0]
                        res_coords = res_coords - min_coords
                        mask = (res_coords < 0).sum(1) == 0
                        res_coords = res_coords[mask]
                        labels = labels[mask]
                        # res_coords = res_coords[:, [2, 1, 0]]

                    for label, coord in zip(labels, res_coords):
                        startc = np.maximum(coord - (cube_size // 2), np.zeros_like(coord))  # noqa
                        startc = startc.astype(int)
                        endc = startc + cube_size
                        # endc = np.minimum(startc + cube_size, label_shape)
                        label_cube_size = endc - startc
                        try:
                            cube = draw_cube(
                                cube_size // 2 - annotation_size // 2,  # top-left edge of label  # noqa
                                annotation_size,  # label size
                                label_cube_size,  # volume shape
                                dtype=dtype,
                                label=label)
                        except:
                            import pdb;pdb.set_trace()

                        # Check if the annotation size fits into the volume
                        vol = volume[startc[0]: endc[0],startc[1]: endc[1],startc[2]: endc[2]]
                        vol_shape = np.asarray(vol.shape[:-1])
                        if np.all(self.cube_size == vol_shape):
                            label_vol[
                                startc[0]: endc[0],
                                startc[1]: endc[1],
                                startc[2]: endc[2]] = np.maximum(
                                    cube,
                                    label_vol[
                                        startc[0]: endc[0],
                                        startc[1]: endc[1],
                                        startc[2]: endc[2]])
                            filtered_coords.append(coord)
                    labels_max = labels.max()
                    label_vol_max = label_vol.max()
                    assert labels_max == label_vol_max, \
                        "label_vol {} and labels {} have different max values.".format(label_vol_max, labels_max)

                    # Now build lists of vols/labels
                    for coord in filtered_coords:
                        startc = np.maximum(coord - (cube_size // 2), np.zeros_like(coord))  # noqa
                        startc = startc.astype(int)
                        endc = np.minimum(startc + cube_size, label_shape)
                        vol = volume[startc[0]: endc[0],startc[1]: endc[1],startc[2]: endc[2]]
                        cube = label_vol[startc[0]: endc[0],startc[1]: endc[1],startc[2]: endc[2]]
                        volume_list.append(vol)
                        label_list.append(cube)

                    # Transpose images if requested
                    volume = np.asarray(volume_list)
                    label = np.asarray(label_list)[..., None]
                    # if self.image_transpose_xyz_zyx:
                    #     volume = volume.transpose(
                    #         self.image_transpose_xyz_zyx)

                    # if self.label_transpose_xyz_zyx:
                    #     label_vol = label_vol.transpose(
                    #         self.label_transpose_xyz_zyx)

                    # # Downsample images if requested.
                    # if self.label_downsample:
                    #     label_vol = resize(
                    #         label_vol,
                    #         self.label_downsample,
                    #         anti_aliasing=True,
                    #         preserve_range=True,
                    #         order=1).astype(dtype)
                    # if self.image_downsample:
                    #     res_volume = []
                    #     res_volume = Parallel(n_jobs=-1)(
                    #         delayed(
                    #             lambda x, y: resize(
                    #                 x,
                    #                 y,
                    #                 anti_aliasing=True,
                    #                 preserve_range=True,
                    #                 order=True))(vol, label_vol.shape[1:]) for vol in tqdm(  # noqa
                    #             volume,
                    #             "Resizing images",
                    #             total=len(volume)))
                    #     volume = np.asarray(res_volume)
                    #     volume = volume.transpose(3, 0, 1, 2)  # Channels first
                    volume = volume.transpose(0, 4, 1, 2, 3)
                    label = label.transpose(0, 4, 1, 2, 3)
                    # label = label[:, None]  # Add singleton channel
                    return volume, label

                elif self.annotation_type == "volumetric":
                    # These annotations are volumetric, for semantic seg.
                    dataset = wk.Dataset(new_dataset_name, scale=list(self.scale))  # noqa
                    if self.source_volume_name:
                        annotation_layer = annotation.save_volume_annotation(dataset, source_volume_name=self.source_volume_name)
                    else:
                        annotation_layer = annotation.save_volume_annotation(dataset)  # noqa

                    if self.bounding_box:  # self.volume_size:
                        # Get the bbox annotation
                        self.bounding_box = np.asarray(self.bounding_box)
                        annotation_layer.bounding_box = BoundingBox(self.bounding_box[0], self.bounding_box[1])
                        # self.volume_size[0], self.volume_size[1])

                    label = annotation_layer.mags[wk.Mag(1)].get_view().read().squeeze(0)
                    label = label.transpose(2, 0, 1)
                    if self.keep_labels:
                        uni_labels = np.unique(label)
                        remap_to_0 = {}
                        for l in uni_labels:
                            if l not in self.keep_labels.keys():
                                remap_to_0[l] = 0
                        keep_labels = dict(self.keep_labels)
                        remap_to_0.update(keep_labels)
                        label = fastremap.remap(label, remap_to_0, preserve_missing_labels=True)

                    # # Then get the dataset images
                    # ims = wk.download_dataset(
                    #     original_dataset_name,
                    #     original_dataset_org,
                    #     bbox=annotation_layer.bounding_box,
                    #     layers=[self.image_layer_name],  # , "Volume Layer"],
                    #     mags=[Mag("1")],
                    #     path="../{}".format(new_dataset_name),
                    # )
                    # image_layer = ims.get_layer(self.image_layer_name)
                    # image_mag = image_layer.get_mag(Mag("1"))
                    # volume = image_mag.read().squeeze(0)
                    volume = read_gcs(self.image_path)

                    # Downsample images if requested.
                    # from matplotlib import pyplot as plt
                    # fn = "tmp.png"
                    # f = plt.figure(figsize=(10, 10))
                    # plt.subplot(121)
                    # imn = 32
                    # plt.imshow(volume[0, imn], cmap="Greys_r")
                    # plt.subplot(122)
                    # plt.imshow(label[imn], cmap="Greys_r")
                    # plt.savefig(fn)
                    # plt.close(f)
                    # path = os.path.join(os.getcwd(), fn)
                    # cmd = "curl --upload-file {} https://transfer.sh/{}".format(path, fn)
                    # _ = os.system(cmd)


                    if self.label_downsample and not np.all(np.asarray(self.label_downsample) == 1):
                        # label = resize(
                        #     label,
                        #     self.label_downsample,
                        #     anti_aliasing=True,
                        #     preserve_range=True,
                        #     order=1).astype(dtype)
                        res_label_shape = np.asarray(label.shape) * self.label_downsample
                        res_label_shape = res_label_shape.astype(int)
                        dtype = label.dtype
                        label = label.astype(np.float32)
                        res_label = Parallel(n_jobs=-1, prefer="threads")(
                            delayed(
                                lambda x, y: resize(
                                    x,
                                    y,
                                    anti_aliasing=False,
                                    preserve_range=True,
                                    order=0))(lab, res_label_shape[1:]) for lab in tqdm(  # noqa
                                label,
                                "Resizing labels",
                                total=len(label)))
                        label = np.asarray(res_label).astype(dtype)

                    if self.image_downsample and not np.all(np.asarray(self.image_downsample) == 1):
                        res_volume = []
                        res_volume = Parallel(n_jobs=-1, prefer="threads")(
                            delayed(
                                lambda x, y: resize(
                                    x,
                                    y,
                                    anti_aliasing=True,
                                    preserve_range=True,
                                    order=1))(vol, label.shape[1:]) for vol in tqdm(  # noqa
                                volume,
                                "Resizing images",
                                total=len(volume)))
                        volume = np.asarray(res_volume)
                        volume = volume.transpose(3, 0, 2, 1)  # Channels first

                    # if self.bounding_box is not None:
                    #     # Crop the labels
                    #     if self.label_downsample:
                    #         res_bounding_box = np.asarray(self.bounding_box) * np.asarray(self.label_downsample)[None]  # noqa
                    #         res_bounding_box = np.floor(res_bounding_box).astype(int)[:, [0, 2, 1]]
                    #     else:
                    #         res_bounding_box = np.asarray(self.bounding_box).astype(int)[:, [0, 2, 1]]
                    #     volume = volume[
                    #         res_bounding_box[0][0]: res_bounding_box[0][0] + res_bounding_box[1][0],
                    #         res_bounding_box[0][1]: res_bounding_box[0][1] + res_bounding_box[1][1],
                    #         res_bounding_box[0][2]: res_bounding_box[0][2] + res_bounding_box[1][2]]
                    #     label = label[
                    #         res_bounding_box[0][0]: res_bounding_box[0][0] + res_bounding_box[1][0],
                    #         res_bounding_box[0][1]: res_bounding_box[0][1] + res_bounding_box[1][1],
                    #         res_bounding_box[0][2]: res_bounding_box[0][2] + res_bounding_box[1][2]]

                    # Split volume/label into cubes then transpose
                    if self.image_transpose_xyz_zyx:
                        volume = volume.transpose(self.image_transpose_xyz_zyx)

                    # Match # z slices between volume and label
                    zslices = min(volume.shape[1], label.shape[0])
                    label = label[:zslices]
                    volume = volume[:, :zslices]

                    # Match H/W between volume and label
                    vh, vw = volume.shape[-2:]
                    lh, lw = label.shape[-2:]
                    volume = volume[..., :lh, :lw]
                    label = label[..., :vh, :vw]

                    # Cut into subvolumes if requested
                    if self.create_subvolumes is not None:
                        dt = ndimage.distance_transform_edt(
                            (label > 0).astype(np.float32)).astype(np.float32)
                        state = np.random.get_state()
                        np.random.seed(42)
                        idxs = skimage.feature.peak_local_max(
                            dt + np.random.random(dt.shape) * 1e-4,
                            indices=True, min_distance=3, threshold_abs=0, threshold_rel=0)
                        np.random.set_state(state)

                        # Now package up label/vol per idx
                        volumes, labels = [], []
                        import pdb;pdb.set_trace()
                        offsets = np.asarray(cube_size) // 2
                        for idx in idxs:
                            z, y, x = idx
                            vol = volume[:,
                                z - offsets[0]: z + offsets[0],
                                y - offsets[1]: y + offsets[1],
                                x - offsets[2]: x + offsets[2]]
                            lab = label[
                                z - offsets[0]: z + offsets[0],
                                y - offsets[1]: y + offsets[1],
                                x - offsets[2]: x + offsets[2]]
                            volumes.append(vol)
                            labels.append(lab)
                        import pdb;pdb.set_trace()
                        volume = np.asarray(volumes)
                        label = np.asarray(labels)
                        del volumes, labels

                        from matplotlib import pyplot as plt
                        fn = "tmp.png"
                        f = plt.figure(figsize=(10, 10))
                        plt.subplot(121)
                        imn = 32
                        plt.imshow(volume[0, 0, imn], cmap="Greys_r")
                        plt.subplot(122)
                        plt.imshow(label[0, imn], cmap="Greys_r")
                        plt.savefig(fn)
                        plt.close(f)
                        path = os.path.join(os.getcwd(), fn)
                        cmd = "curl --upload-file {} https://transfer.sh/{}".format(path, fn)
                        _ = os.system(cmd)
                    else:
                        # Add dims for handling data on tpus
                        volume = volume[None]
                        label = label[None, None].astype(np.uint8)
                    return volume, label
