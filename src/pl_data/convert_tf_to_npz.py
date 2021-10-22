import os
import numpy as np
import tensorflow as tf
import sys
from tqdm import tqdm
os.environ['NO_GCE_CHECK'] = 'true'
# Run the following beforehand
# gcloud login application-default


def create_file(self, filename):
  """Create a file.

  The retry_params specified in the open call will override the default
  retry params for this particular file handle.

  Args:
    filename: filename.
  """
  self.response.write('Creating file %s\n' % filename)

  write_retry_params = gcs.RetryParams(backoff_factor=1.1)
  gcs_file = gcs.open(filename,
                      'w',
                      content_type='text/plain',
                      options={'x-goog-meta-foo': 'foo',
                               'x-goog-meta-bar': 'bar'},
                      retry_params=write_retry_params)
  gcs_file.write('abcde\n')
  gcs_file.write('f'*1024*4 + '\n')
  gcs_file.close()
  self.tmp_filenames_to_clean_up.append(filename)


def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


fold = "val"
# output_dir = "gs://serrelab/connectomics/npzs/celltype/{}".format(fold)
output_dir = "temp_npz/celltype/{}".format(fold)
os.makedirs(output_dir, exist_ok=True)
filename = "gs://serrelab/connectomics/tfrecords/celltype/cell_type_10_64_15.tfrecords_{}.tfrecords".format(fold)

raw_dataset = tf.data.TFRecordDataset(os.path.join(filename))
image_feature_description = {
    'volume': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
}
parsed_image_dataset = raw_dataset.map(_parse_image_function)
for idx, image_features in tqdm(enumerate(parsed_image_dataset)):
    volume_bytes = tf.reshape(image_features['volume'], shape=[])
    label_bytes = tf.reshape(image_features['label'], shape=[])
    volume = tf.io.decode_raw(volume_bytes, tf.float32)
    volume = tf.reshape(volume, [64, 128, 128, 2])
    label = tf.io.decode_raw(label_bytes, tf.float32)
    label = tf.reshape(label, [64, 128, 128, 6])

    # Transpose now for pytorch
    volume = volume.numpy().transpose(3, 0, 1, 2)
    label = label.numpy().transpose(3, 0, 1, 2)

    path = os.path.join(output_dir, "{}".format(idx))
    np.savez(path, volume=volume, label=label)

