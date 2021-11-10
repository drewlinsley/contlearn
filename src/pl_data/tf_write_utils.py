import os
import tensorflow as tf
import sys
from google.cloud import storage


def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


output_dir = "npz_files"
# filename = "clickme_train"
filename = sys.argv[1]
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, filename), exist_ok=True)

raw_dataset = tf.data.TFRecordDataset(os.path.join("archive", "{}.tfrecords".format(filename)))
image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'click_count': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
    'heatmap': tf.io.FixedLenFeature([], tf.string),
}
parsed_image_dataset = raw_dataset.map(_parse_image_function)
for idx, image_features in enumerate(parsed_image_dataset):
    image_bytes = tf.reshape(image_features['image'], shape=[])
    hm_bytes = tf.reshape(image_features['heatmap'], shape=[])
    label = image_features["label"].numpy()
    clicks = image_features["click_count"].numpy()
    image = tf.io.decode_raw(image_bytes, tf.float32)
    image = tf.reshape(image, [256, 256, 3])
    heatmap = tf.io.decode_raw(hm_bytes, tf.float32)
    heatmap = tf.reshape(heatmap, [256, 256])
    path = os.path.join(output_dir, filename, "{}".format(idx))
    np.savez(path, image=image, heatmap=heatmap, clicks=clicks, label=label)
