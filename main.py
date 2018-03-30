"""Code for training CycleGAN."""
import os
import argparse
import random
import numpy as np
from scipy.misc import imsave
import tensorflow as tf
import utils
import model


class CycleGAN(object):
  """The CycleGAN module."""

  def __init__(self, exp_name, restore, base_lr,
               mode, max_step, dataset_name,
               weight_dir, flipping, skip):

    self._pool_size = utils.POOL_SIZE
    self._exp_name = os.path.join(exp_name)
    self._sample_dir = os.path.join(utils.SAMPLE_DIR, self._exp_name)
    self._log_dir = os.path.join(utils.LOG_DIR, self._exp_name)
    self._ckpt_dir = os.path.join(utils.CKPT_DIR, self._exp_name)
    self._num_imgs_to_save = utils.NUM_IMG_TO_SAVE
    self._restore = restore
    self._base_lr = base_lr
    self._max_step = max_step
    self._dataset_name = dataset_name
    self._mode = mode
    self._weight_dir = weight_dir if weight_dir else self._ckpt_dir
    self._flipping = flipping
    self._skip = skip

    self.fake_a_pool = np.zeros(
        (self._pool_size, 1, utils.IMG_HEIGHT, utils.IMG_WIDTH,
         utils.IMG_CHANNEL)
    )
    self.fake_b_pool = np.zeros(
        (self._pool_size, 1, utils.IMG_HEIGHT, utils.IMG_WIDTH,
         utils.IMG_CHANNEL)
    )

  def model_setup(self):
    """
    This function sets up the model to train.

    self.input_a/self.input_b -> Set of training images.
    self.fake_a/self.fake_b -> Generated images by corresponding generator
    of input_a and input_b
    self.lr -> Learning rate variable
    self.cyc_a/ self.cyc_b -> Images generated after feeding
    self.fake_a/self.fake_b to corresponding generator.
    This is use to calculate cyclic loss
    """
    self.input_a = tf.placeholder(
        tf.float32, [
            utils.BATCH_SIZE,
            utils.IMG_WIDTH,
            utils.IMG_HEIGHT,
            utils.IMG_CHANNEL
        ], name="input_a")
    self.input_b = tf.placeholder(
        tf.float32, [
            utils.BATCH_SIZE,
            utils.IMG_WIDTH,
            utils.IMG_HEIGHT,
            utils.IMG_CHANNEL
        ], name="input_b")

    self.fake_pool_a = tf.placeholder(
        tf.float32, [
            None,
            utils.IMG_WIDTH,
            utils.IMG_HEIGHT,
            utils.IMG_CHANNEL
        ], name="fake_pool_a")
    self.fake_pool_b = tf.placeholder(
        tf.float32, [
            None,
            utils.IMG_WIDTH,
            utils.IMG_HEIGHT,
            utils.IMG_CHANNEL
        ], name="fake_pool_b")

    # self.global_step = tf.contrib.slim.get_or_create_global_step()
    self.global_step = tf.train.get_or_create_global_step()

    self.num_fake_inputs = 0

    self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

    inputs = {
        'images_a': self.input_a,
        'images_b': self.input_b,
        'fake_pool_a': self.fake_pool_a,
        'fake_pool_b': self.fake_pool_b,
    }

    outputs = model.get_outputs(inputs, skip=self._skip)

    self.prob_real_a_is_real = outputs['prob_real_a_is_real']
    self.prob_real_b_is_real = outputs['prob_real_b_is_real']
    self.fake_images_a = outputs['fake_images_a']
    self.fake_images_b = outputs['fake_images_b']
    self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
    self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

    self.cycle_images_a = outputs['cycle_images_a']
    self.cycle_images_b = outputs['cycle_images_b']

    self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
    self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']

  def compute_losses(self):
    """
    In this function we are defining the variables for loss calculations
    and training model.

    d_loss-a/d_loss_b -> loss for discriminator A/B
    g_loss-a/g_loss_b -> loss for generator A/B
    *_trainer -> Various trainer for above loss functions
    *_summ -> Summary variables for above loss functions
    """
    cycle_consistency_loss_a = utils.LAMBDA_A * utils.cycle_consistency_loss(
        real_images=self.input_a, generated_images=self.cycle_images_a)
    cycle_consistency_loss_b = utils.LAMBDA_B * utils.cycle_consistency_loss(
        real_images=self.input_b, generated_images=self.cycle_images_b)

    lsgan_loss_a = utils.lsgan_loss_generator(self.prob_fake_a_is_real)
    lsgan_loss_b = utils.lsgan_loss_generator(self.prob_fake_b_is_real)

    self.gen_a_loss = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b
    self.gen_b_loss = cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a

    self.dis_a_loss = utils.lsgan_loss_discriminator(
        prob_real_is_real=self.prob_real_a_is_real,
        prob_fake_is_real=self.prob_fake_pool_a_is_real)
    self.dis_b_loss = utils.lsgan_loss_discriminator(
        prob_real_is_real=self.prob_real_b_is_real,
        prob_fake_is_real=self.prob_fake_pool_b_is_real)

    optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

    self.model_vars = tf.trainable_variables()

    dis_a_vars = [var for var in self.model_vars if 'discriminator_a' in var.name]
    gen_a_vars = [var for var in self.model_vars if 'generator_a' in var.name]
    dis_b_vars = [var for var in self.model_vars if 'discriminator_b' in var.name]
    gen_b_vars = [var for var in self.model_vars if 'generator_b' in var.name]

    self.dis_a_trainer = optimizer.minimize(self.dis_a_loss, var_list=dis_a_vars)
    self.dis_b_trainer = optimizer.minimize(self.dis_b_loss, var_list=dis_b_vars)
    self.gen_a_trainer = optimizer.minimize(self.gen_a_loss, var_list=gen_a_vars)
    self.gen_b_trainer = optimizer.minimize(self.gen_b_loss, var_list=gen_b_vars)

    for var in self.model_vars:
      print(var.name)

    # Summary variables for tensorboard
    self.gen_a_loss_summ = tf.summary.scalar("gen_a_loss", self.gen_a_loss)
    self.gen_b_loss_summ = tf.summary.scalar("gen_b_loss", self.gen_b_loss)
    self.dis_a_loss_summ = tf.summary.scalar("dis_a_loss", self.dis_a_loss)
    self.dis_b_loss_summ = tf.summary.scalar("dis_b_loss", self.dis_b_loss)

  def save_images(self, sess, epoch):
    """
    Saves input and output images.
    :param sess: The session.
    :param epoch: Currnt epoch.
    """
    if not os.path.exists(self._sample_dir):
      os.makedirs(self._sample_dir)

    names = ['input_a_', 'input_b_', 'fake_a_',
             'fake_b_', 'cyc_a_', 'cyc_b_']

    with open(os.path.join(self._sample_dir,
                           'epoch_' + str(epoch) + '.html'), 'w') as v_html:
      for i in range(0, self._num_imgs_to_save):
        print("Saving image {}/{}".format(i, self._num_imgs_to_save))
        inputs = sess.run(self.inputs)
        fake_a_temp, fake_b_temp, cyc_a_temp, cyc_b_temp = sess.run([
            self.fake_images_a,
            self.fake_images_b,
            self.cycle_images_a,
            self.cycle_images_b
        ], feed_dict={
            self.input_a: inputs['images_i'],
            self.input_b: inputs['images_j']
        })

        tensors = [inputs['images_i'], inputs['images_j'],
                   fake_b_temp, fake_a_temp, cyc_a_temp, cyc_b_temp]

        for name, tensor in zip(names, tensors):
          image_name = name + str(epoch) + "_" + str(i) + ".jpg"
          imsave(os.path.join(self._sample_dir, image_name),
                 ((tensor[0] + 1) * 127.5).astype(np.uint8))
          v_html.write("<img src=\"" + image_name + "\">")
        v_html.write("<br>")

  def fake_image_pool(self, num_fakes, fake, fake_pool):
    """
    This function saves the generated image to corresponding
    pool of images.

    It keeps on filling the pool till it is full and then randomly
    selects an already stored image and replace it with new one.
    """
    if num_fakes < self._pool_size:
      fake_pool[num_fakes] = fake
      return fake
    else:
      if random.random() > 0.5:
        random_id = random.randint(0, self._pool_size - 1)
        temp = fake_pool[random_id]
        fake_pool[random_id] = fake
        return temp
      else:
        return fake

  def train(self):
    """Training Function."""
    # Load Dataset from the dataset folder
    self.inputs = utils.load_data(
        self._dataset_name, utils.SIZE_BEFORE_CROP,
        self._mode, True, self._flipping)

    # Build the network
    self.model_setup()

    # Loss function calculations
    self.compute_losses()

    # Initializing the global variables
    init = (tf.global_variables_initializer(),
            tf.local_variables_initializer())
    saver = tf.train.Saver()

    max_image_num = utils.get_data_size(self._dataset_name, self._mode)

    with tf.Session() as sess:
      sess.run(init)

      # Restore the model to run the model from last checkpoint
      if self._restore:
        ckpt_fname = tf.train.latest_checkpoint(self._weight_dir)
        saver.restore(sess, ckpt_fname)

      if not os.path.exists(self._log_dir):
        os.makedirs(self._log_dir)

      writer = tf.summary.FileWriter(self._log_dir)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      # global_step = tf.contrib.slim.get_or_create_global_step()

      # Training Loop
      for epoch in range(sess.run(self.global_step), self._max_step):
        print("In the epoch ", epoch)
        saver.save(sess, self._ckpt_dir, global_step=epoch)

        # Dealing with the learning rate as per the epoch number
        if epoch < 100:
          curr_lr = self._base_lr
        else:
          curr_lr = self._base_lr - self._base_lr * (epoch - 100) / 100

        self.save_images(sess, epoch)

        for i in range(0, max_image_num):

          inputs = sess.run(self.inputs)

          # Optimizing the G_A network
          _, fake_b_temp, ga_loss, summary_str = sess.run(
              [self.gen_a_trainer,
               self.fake_images_b,
               self.gen_a_loss,
               self.gen_a_loss_summ],
              feed_dict={
                  self.input_a: inputs['images_i'],
                  self.input_b: inputs['images_j'],
                  self.learning_rate: curr_lr
              }
          )
          writer.add_summary(summary_str, epoch * max_image_num + i)

          fake_b_from_pool = self.fake_image_pool(
              self.num_fake_inputs, fake_b_temp, self.fake_b_pool)

          # Optimizing the D_B network
          _, db_loss, summary_str = sess.run(
              [self.dis_b_trainer,
               self.dis_b_loss,
               self.dis_b_loss_summ],
              feed_dict={
                  self.input_a: inputs['images_i'],
                  self.input_b: inputs['images_j'],
                  self.learning_rate: curr_lr,
                  self.fake_pool_b: fake_b_from_pool
              }
          )
          writer.add_summary(summary_str, epoch * max_image_num + i)

          # Optimizing the G_B network
          _, fake_a_temp, gb_loss, summary_str = sess.run(
              [self.gen_b_trainer,
               self.fake_images_a,
               self.gen_b_loss,
               self.gen_b_loss_summ],
              feed_dict={
                  self.input_a: inputs['images_i'],
                  self.input_b: inputs['images_j'],
                  self.learning_rate: curr_lr
              }
          )
          writer.add_summary(summary_str, epoch * max_image_num + i)

          fake_a_from_pool = self.fake_image_pool(
              self.num_fake_inputs, fake_a_temp, self.fake_a_pool)

          # Optimizing the D_A network
          _, da_loss, summary_str = sess.run(
              [self.dis_a_trainer,
               self.dis_a_loss,
               self.dis_a_loss_summ],
              feed_dict={
                  self.input_a: inputs['images_i'],
                  self.input_b: inputs['images_j'],
                  self.learning_rate: curr_lr,
                  self.fake_pool_a: fake_a_from_pool
              }
          )
          writer.add_summary(summary_str, epoch * max_image_num + i)

          writer.flush()
          self.num_fake_inputs += 1
          print('Epoch: %3d, Batch %4d/%d, GA:%.6f, DA:%.6f GB:%.6f, DB:%.6f' %
                (epoch, i, max_image_num, ga_loss, da_loss, gb_loss, db_loss))

        sess.run(tf.assign(self.global_step, epoch + 1))

      coord.request_stop()
      coord.join(threads)
      writer.add_graph(sess.graph)

  def test(self):
    """Test Function."""
    print("Testing the results")

    self.inputs = utils.load_data(
        self._dataset_name, utils.SIZE_BEFORE_CROP,
        self._mode, False, self._flipping)

    self.model_setup()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init)

      ckpt_fname = tf.train.latest_checkpoint(self._weight_dir)
      saver.restore(sess, ckpt_fname)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      self._num_imgs_to_save = utils.get_data_size(self._dataset_name, self._mode)
      self.save_images(sess, 0)

      coord.request_stop()
      coord.join(threads)


def main(args):
  """
  :param to_train: Specify whether it is training or testing. 1: training; 2:
   resuming from latest checkpoint; 0: testing.
  :param log_dir: The root dir to save checkpoints and imgs. The actual dir
  is the root dir appended by the folder with the name timestamp.
  """
  assert args.dataset_name is not None, 'Please Specify Dataset Name!'

  if not os.path.isdir(utils.LOG_DIR):
    os.makedirs(utils.LOG_DIR)
  if not os.path.isdir(utils.CKPT_DIR):
    os.makedirs(utils.CKPT_DIR)
  restore = ((args.mode == 'train') and (args.weight_dir is not None))
  exp_name = args.dataset_name if args.exp_name is None\
      else args.dataset_name + '_' + args.exp_name
  cyclegan_model = CycleGAN(exp_name, restore, args.base_lr, args.mode,
                            args.max_step, args.dataset_name,
                            args.weight_dir, args.flipping, args.skip)
  if args.mode == 'train':
    cyclegan_model.train()
  else:
    cyclegan_model.test()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, choices=['train', 'test'],
                      default='train', help='train or test mode.')
  parser.add_argument('-n', '--exp_name', type=str,
                      default=None, help='the experiments name add to log & ckpt dir')
  parser.add_argument('-w', '--weight_dir', type=str,
                      default=None, help='The directory of checkpoints that wants to load.')
  parser.add_argument('-s', '--skip', type=bool,
                      default=True, help='Whether to add skip connection between input and output.')
  parser.add_argument('-l', '--base_lr', type=float,
                      default=2e-4, help='base learning rate, will automatic decrease')
  parser.add_argument('-e', '--max_step', type=int,
                      default=200, help='how many step(epoch)s to train')
  parser.add_argument('-d', '--dataset_name', type=str,
                      default=None, help='which dataset to train')
  parser.add_argument('-f', '--flipping', type=bool,
                      default=True, help='flipping images or not')
  ARGS = parser.parse_args()
  print(ARGS)
  main(ARGS)
