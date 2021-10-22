import os
import tensorflow as tf
import sys
os.environ['NO_GCE_CHECK'] = 'true'


def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


output_dir = "gs://serrelab/connectomics/npzs/celltype/train"
filename = "gs://serrelab/connectomics/tfrecords/celltype/cell_type_10_64_15.tfrecords_train.tfrecords"

raw_dataset = tf.data.TFRecordDataset(os.path.join(filename))
image_feature_description = {
    'volume': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
}
parsed_image_dataset = raw_dataset.map(_parse_image_function)
for idx, image_features in enumerate(parsed_image_dataset):
    volume_bytes = tf.reshape(image_features['volume'], shape=[])
    label_bytes = tf.reshape(image_features['label'], shape=[])
    volume = tf.io.decode_raw(volume_bytes, tf.float32)
    volume = tf.reshape(volume, [64, 128, 128, 2])
    label = tf.io.decode_raw(label_bytes, tf.float32)
    label = tf.reshape(label, [64, 128, 128, 2])

    # Transpose now for pytorch
    volume = volume.transpose(3, 0, 1, 2)
    label = label.transpose(3, 0, 1, 2)

    path = os.path.join(output_dir, filename, "{}".format(idx))
    np.savez(path, volume=volume, label=label)

