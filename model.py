"""Code for constructing the model and get the outputs from the model."""

import tensorflow as tf
from layers import conv2d, conv2d_trans
import utils


NUM_GEN_FLITER = 32
NUM_DIS_FLITER = 64


def get_outputs(inputs, skip=False):
  images_a = inputs['images_a']
  images_b = inputs['images_b']

  fake_pool_a = inputs['fake_pool_a']
  fake_pool_b = inputs['fake_pool_b']

  with tf.variable_scope('Model') as scope:

    prob_real_a_is_real = discriminator(images_a, 'discriminator_a')
    prob_real_b_is_real = discriminator(images_b, 'discriminator_b')

    fake_images_b = generator(images_a, 'generator_a', skip=skip)
    fake_images_a = generator(images_b, 'generator_b', skip=skip)

    scope.reuse_variables()

    prob_fake_a_is_real = discriminator(fake_images_a, 'discriminator_a')
    prob_fake_b_is_real = discriminator(fake_images_b, 'discriminator_b')

    cycle_images_a = generator(fake_images_b, 'generator_b', skip=skip)
    cycle_images_b = generator(fake_images_a, 'generator_a', skip=skip)

    scope.reuse_variables()

    prob_fake_pool_a_is_real = discriminator(fake_pool_a, 'discriminator_a')
    prob_fake_pool_b_is_real = discriminator(fake_pool_b, 'discriminator_b')

  return {
      'prob_real_a_is_real': prob_real_a_is_real,
      'prob_real_b_is_real': prob_real_b_is_real,
      'prob_fake_a_is_real': prob_fake_a_is_real,
      'prob_fake_b_is_real': prob_fake_b_is_real,
      'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
      'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
      'cycle_images_a': cycle_images_a,
      'cycle_images_b': cycle_images_b,
      'fake_images_a': fake_images_a,
      'fake_images_b': fake_images_b,
  }


def build_resnet_block(inputs, out_dim, name='resnet', padding='REFLECT'):
  """build a single block of resnet.
  :param inputres: inputs
  :param out_dim: output dim
  :param name: name
  :param padding: for tensorflow version use REFLECT; for pytorch version use
   CONSTANT
  :return: a single block of resnet.
  """
  kernel = 3
  stride = 1
  with tf.variable_scope(name):
    outputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
    outputs = conv2d(outputs, out_dim, kernel, stride,
                     padding='VALID', name='conv1')
    outputs = tf.pad(outputs, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
    outputs = conv2d(outputs, out_dim, kernel, stride,
                     padding='VALID', name='conv2', activation=False)
    return tf.nn.relu(outputs + inputs)


def generator(inputs, name='generator', skip=False, padding='REFLECT'):
  with tf.variable_scope(name):
    end_kernel = 7
    kernel = 3

    pad_input = tf.pad(inputs,
                       [[0, 0], [kernel, kernel], [kernel, kernel], [0, 0]],
                       padding)
    o_c1 = conv2d(pad_input, NUM_GEN_FLITER, end_kernel, 1, name='conv1')
    o_c2 = conv2d(o_c1, NUM_GEN_FLITER * 2, kernel, 2, padding='SAME', name='conv2')
    o_c3 = conv2d(o_c2, NUM_GEN_FLITER * 4, kernel, 2, padding='SAME', name='conv3')

    o_r1 = build_resnet_block(o_c3, NUM_GEN_FLITER * 4, 'res1', padding)
    o_r2 = build_resnet_block(o_r1, NUM_GEN_FLITER * 4, 'res2', padding)
    o_r3 = build_resnet_block(o_r2, NUM_GEN_FLITER * 4, 'res3', padding)
    o_r4 = build_resnet_block(o_r3, NUM_GEN_FLITER * 4, 'res4', padding)
    o_r5 = build_resnet_block(o_r4, NUM_GEN_FLITER * 4, 'res5', padding)
    o_r6 = build_resnet_block(o_r5, NUM_GEN_FLITER * 4, 'res6', padding)
    o_r7 = build_resnet_block(o_r6, NUM_GEN_FLITER * 4, 'res7', padding)
    o_r8 = build_resnet_block(o_r7, NUM_GEN_FLITER * 4, 'res8', padding)
    o_r9 = build_resnet_block(o_r8, NUM_GEN_FLITER * 4, 'res9', padding)

    o_c4 = conv2d_trans(o_r9, [utils.BATCH_SIZE, 128, 128, NUM_GEN_FLITER * 2],
                        NUM_GEN_FLITER * 2, kernel, 2, padding='SAME', name='conv4')
    o_c5 = conv2d_trans(o_c4, [utils.BATCH_SIZE, 256, 256, NUM_GEN_FLITER],
                        NUM_GEN_FLITER, kernel, 2, padding='SAME', name='conv5')
    o_c6 = conv2d(o_c5, utils.IMG_CHANNEL, end_kernel, 1, padding='SAME', name='conv6',
                  normalize=False, activation=False)

    if skip is True:
      outputs = tf.nn.tanh(inputs + o_c6, 'tanh')
    else:
      outputs = tf.nn.tanh(o_c6, 'tanh')

    return outputs


def discriminator(inputs, name='discriminator'):
  with tf.variable_scope(name):
    kernel = 4

    o_c1 = conv2d(inputs, NUM_DIS_FLITER, kernel, 2,
                  padding='SAME', name='conv1',
                  normalize=False, relu_factor=0.2)
    o_c2 = conv2d(o_c1, NUM_DIS_FLITER * 2, kernel, 2,
                  padding='SAME', name='conv2', relu_factor=0.2)
    o_c3 = conv2d(o_c2, NUM_DIS_FLITER * 4, kernel, 2,
                  padding='SAME', name='conv3', relu_factor=0.2)
    o_c4 = conv2d(o_c3, NUM_DIS_FLITER * 8, kernel, 1,
                  padding='SAME', name='conv4', relu_factor=0.2)
    o_c5 = conv2d(o_c4, 1, kernel, 1, padding='SAME', name='conv5',
                  normalize=False, activation=False)

    return o_c5


def patch_discriminator(inputs, name='discriminator'):
  with tf.variable_scope(name):
    kernel = 4

    patch_input = tf.random_crop(inputs, [1, 70, 70, 3])
    o_c1 = conv2d(patch_input, NUM_DIS_FLITER, kernel, 2,
                  padding='SAME', name='c1', normalize=False,
                  relu_factor=0.2)
    o_c2 = conv2d(o_c1, NUM_DIS_FLITER * 2, kernel, 2,
                  padding='SAME', name='c2', relu_factor=0.2)
    o_c3 = conv2d(o_c2, NUM_DIS_FLITER * 4, kernel, 2,
                  padding='SAME', name='c3', relu_factor=0.2)
    o_c4 = conv2d(o_c3, NUM_DIS_FLITER * 8, kernel, 2,
                  padding='SAME', name='c4', relu_factor=0.2)
    o_c5 = conv2d(o_c4, 1, kernel, 1, padding='SAME', name='c5',
                  normalize=False, activation=False)

    return o_c5
