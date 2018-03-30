import tensorflow as tf


def lrelu(inputs, leak=0.2, name="lrelu"):
  with tf.variable_scope(name):
    return tf.maximum(inputs, leak * inputs)


def instance_norm(inputs):
  with tf.variable_scope("instance_norm"):
    epsilon = 1e-5
    mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
    scale = tf.get_variable('scale', [inputs.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(
                                mean=1.0, stddev=0.02))
    offset = tf.get_variable(
        'offset', [inputs.get_shape()[-1]],
        initializer=tf.constant_initializer(0.0)
    )
    out = scale * tf.div(inputs - mean, tf.sqrt(var + epsilon)) + offset
    return out


def conv2d(inputs, out_dim, kernel, stride, stddev=0.02,
           padding="VALID", name="conv2d", normalize=True,
           activation=True, relu_factor=0):
  """
  `convolution` creates a variable called `weights`, representing the
  convolutional kernel, that is convolved (actually cross-correlated) with the
  `inputs` to produce a `Tensor` of activations. If a `biases_initializer` is
  provided then a `biases` variable would be created.
  """
  with tf.variable_scope(name):
    conv = tf.contrib.layers.conv2d(
        inputs, out_dim, kernel, stride, padding, activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
        biases_initializer=tf.constant_initializer(0.0))

    if normalize:
      conv = instance_norm(conv)

    if activation:
      if(relu_factor == 0):
        conv = tf.nn.relu(conv, "relu")
      else:
        conv = lrelu(conv, relu_factor, "lrelu")

    return conv


def conv2d_trans(inputs, outshape, out_dim, kernel, stride, stddev=0.02,
                 padding="VALID", name="conv2d_trans", normalize=True,
                 activation=True, relu_factor=0):
  """
  The function creates a variable called `weights`, representing the
  kernel, that is convolved with the input. And a second variable called
  'biases' is added to the result of the operation.
  """
  if outshape is not None:
    pass
  with tf.variable_scope(name):
    conv = tf.contrib.layers.conv2d_transpose(
        inputs, out_dim, kernel, stride, padding, activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
        biases_initializer=tf.constant_initializer(0.0))

    if normalize:
      conv = instance_norm(conv)
      # conv = tf.contrib.layers.batch_norm(conv, decay=0.9,
      # updates_collections=None, epsilon=1e-5, scale=True,
      # scope="batch_norm")

    if activation:
      if(relu_factor == 0):
        conv = tf.nn.relu(conv, "relu")
      else:
        conv = lrelu(conv, relu_factor, "lrelu")

    return conv
