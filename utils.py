import os
import csv
import tensorflow as tf


LAMBDA_A = float(10)
LAMBDA_B = float(10)

LOG_DIR = 'log'
CKPT_DIR = 'checkpoint'
SAMPLE_DIR = 'sample'

SIZE_BEFORE_CROP = 286  # Resize to this size before random cropping.
NUM_IMG_TO_SAVE = 20

BATCH_SIZE = 1  # The number of samples per batch.
IMG_WIDTH = 256  # The width of each image.
IMG_HEIGHT = 256  # The height of each image.
IMG_CHANNEL = 3  # The number of color channels per image.
POOL_SIZE = 50  # reserve pool size, for storing past generated images

IMG_TYPE = '.jpg'


def cycle_consistency_loss(real_images, generated_images):
  """Compute the cycle consistency loss.
  The cycle consistency loss is defined as the sum of the L1 distances
  between the real images from each domain and their generated (fake)
  counterparts.
  """
  return tf.reduce_mean(tf.abs(real_images - generated_images))


def lsgan_loss_generator(prob_fake_is_real):
  """Computes the LS-GAN loss as minimized by the generator.
  Rather than compute the negative loglikelihood, a least-squares loss is
  used to optimize the discriminators.
  """
  return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))


def lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
  return (tf.reduce_mean(tf.squared_difference(prob_real_is_real, 1)) +
          tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 0))) * 0.5


def get_data_size(dataset_name, mode):
  file_path = os.path.join('./data', dataset_name,
                           '%s_%s.csv' % (dataset_name, mode))
  with open(file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    return len(list(csv_reader))


def _load_samples(csv_name, image_type):
  filename_queue = tf.train.string_input_producer(
      [csv_name])

  reader = tf.TextLineReader()
  _, csv_filename = reader.read(filename_queue)

  record_defaults = [tf.constant([], dtype=tf.string),
                     tf.constant([], dtype=tf.string)]

  filename_i, filename_j = tf.decode_csv(
      csv_filename, record_defaults=record_defaults)

  file_contents_i = tf.read_file(filename_i)
  file_contents_j = tf.read_file(filename_j)
  if image_type == '.jpg':
    image_decoded_a = tf.image.decode_jpeg(
        file_contents_i, channels=IMG_CHANNEL)
    image_decoded_b = tf.image.decode_jpeg(
        file_contents_j, channels=IMG_CHANNEL)
  elif image_type == '.png':
    image_decoded_a = tf.image.decode_png(
        file_contents_i, channels=IMG_CHANNEL, dtype=tf.uint8)
    image_decoded_b = tf.image.decode_png(
        file_contents_j, channels=IMG_CHANNEL, dtype=tf.uint8)

  return image_decoded_a, image_decoded_b


def load_data(dataset_name, image_size_before_crop,
              mode, shuffle=True, flipping=True):
  """
  :param dataset_name: The name of the dataset.
  :param image_size_before_crop: Resize to this size before random cropping.
  :param shuffle: Shuffle switch.
  :param flipping: Flip switch.
  :return:
  """
  csv_name = os.path.join('./data', dataset_name,
                          '%s_%s.csv' % (dataset_name, mode))
  image_a, image_b = _load_samples(csv_name, IMG_TYPE)
  inputs = {'image_a': image_a, 'image_b': image_b}

  # Preprocessing:
  inputs['image_a'] = tf.image.resize_images(
      inputs['image_a'], [image_size_before_crop, image_size_before_crop])
  inputs['image_b'] = tf.image.resize_images(
      inputs['image_b'], [image_size_before_crop, image_size_before_crop])

  if flipping is True:
    inputs['image_a'] = tf.image.random_flip_left_right(inputs['image_a'])
    inputs['image_b'] = tf.image.random_flip_left_right(inputs['image_b'])

  inputs['image_a'] = tf.random_crop(
      inputs['image_a'], [IMG_HEIGHT, IMG_WIDTH, 3])
  inputs['image_b'] = tf.random_crop(
      inputs['image_b'], [IMG_HEIGHT, IMG_WIDTH, 3])

  inputs['image_a'] = tf.subtract(tf.div(inputs['image_a'], 127.5), 1)
  inputs['image_b'] = tf.subtract(tf.div(inputs['image_b'], 127.5), 1)

  # Batch
  if shuffle is True:
    inputs['images_i'], inputs['images_j'] = tf.train.shuffle_batch(
        [inputs['image_a'], inputs['image_b']], 1, 5000, 100)
  else:
    inputs['images_i'], inputs['images_j'] = tf.train.batch(
        [inputs['image_a'], inputs['image_b']], 1)

  return inputs


def time2str(time):
  assert isinstance(time, float), 'input time must a float!'
  if time > 3600:
    time_format = '%d:%02d:%02d' % (
        time // 3600, (time % 3600) // 60, time % 60)
  elif time > 60:
    time_format = '%3d:%02d' % (time // 60, time % 60)
  else:
    time_format = '%5ds' % time
  return time_format
