from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np

from utils import train_utils
import model
import data


import tensorflow as tf

slim = tf.contrib.slim


flags = tf.app.flags

FLAGS = flags.FLAGS


flags.DEFINE_string('train_logdir', './models',
                    'Where the checkpoint and logs are stored.')
flags.DEFINE_string('ckpt_name_to_save', 'resnet_v2_34.ckpt',
                    'Name to save checkpoint file')
flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')
flags.DEFINE_string('summaries_dir', './models/train_logs',
                     'Where to save summary logs for TensorBoard.')

flags.DEFINE_float('base_learning_rate', .0005,
                   'The learning rate for model training.')
flags.DEFINE_float('learning_rate_decay_factor', 1e-2,
                   'The rate to decay the base learning rate.')
flags.DEFINE_integer('learning_rate_decay_step', 6000,
                     'Decay the base learning rate at a fixed step.')
flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')
flags.DEFINE_integer('training_number_of_steps', 300000,
                     'The number of steps used for training')
flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')
flags.DEFINE_integer('slow_start_step', 1000,
                     'Training model with small learning rate for few steps.')
flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

# Settings for training strategy.
flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

# Settings for fine-tuning the network.
flags.DEFINE_string('saved_checkpoint_dir',
                    None,
                    'Saved checkpoint dir.')
flags.DEFINE_string('pre_trained_checkpoint',
                    None,
                    'The pre-trained checkpoint in tensorflow format.')
flags.DEFINE_string('checkpoint_exclude_scopes',
                    None,
                    'Comma-separated list of scopes of variables to exclude '
                    'when restoring from a checkpoint.')
flags.DEFINE_string('trainable_scopes',
                    None,
                    'Comma-separated list of scopes to filter the set of variables '
                    'to train. By default, None would train all the variables.')
flags.DEFINE_string('checkpoint_model_scope',
                    None,
                    'Model scope in the checkpoint. None if the same as the trained model.')
flags.DEFINE_string('model_name',
                    'resnet_v2_34',
                    'The name of the architecture to train.')
flags.DEFINE_boolean('ignore_missing_vars',
                     False,
                     'When restoring a checkpoint would ignore missing variables.')

flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/minc-2500',
                    'Where the dataset reside.')

flags.DEFINE_integer('how_many_training_epochs', 120,
                     'How many training loops to run')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('resize_height', 320, 'resize_height')
flags.DEFINE_integer('resize_width', 320, 'resize_width')
flags.DEFINE_string('labels',
                    'brick,carpet,hair,leather,wallpaper',
                    'Labels to use')


MINC2500_TRAIN_DATA_SIZE = 10625
MINC2500_VALIDATE_DATA_SIZE = 625


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)

    tf.gfile.MakeDirs(FLAGS.train_logdir)
    tf.logging.info('Creating train logdir: %s', FLAGS.train_logdir)

    with tf.Graph().as_default() as graph:
        global_step = tf.train.get_or_create_global_step()

        X = tf.placeholder(tf.float32,
                           [FLAGS.batch_size, FLAGS.resize_height, FLAGS.resize_width, 3],
                           name='X')
        ground_truth = tf.placeholder(tf.int64, [None], name='ground_truth')
        is_training = tf.placeholder(tf.bool, name='is_training')
        # dropout_keep_prob = tf.placeholder(tf.float32)
        # learning_rate = tf.placeholder(tf.float32, [], name='lr')

        logits = model.ten(X, num_classes, is_training, FLAGS.batch_size)

        # Print name and shape of parameter nodes  (values not yet initialized)
        tf.logging.info("++++++++++++++++++++++++++++++++++")
        tf.logging.info("Parameters")
        tf.logging.info("++++++++++++++++++++++++++++++++++")
        for v in slim.get_model_variables():
            tf.logging.info('name = %s, shape = %s' % (v.name, v.get_shape()))

        # Define loss
        tf.losses.sparse_softmax_cross_entropy(labels=ground_truth, logits=logits)

        # Gather update ops. These contain, for example, the updates for the
        # batch_norm variables created by model.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        predition = tf.argmax(logits, 1, name='prediction')
        correct_prediction = tf.equal(predition, ground_truth, name='correct_prediction')
        confusion_matrix = tf.confusion_matrix(ground_truth,
                                               predition,
                                               num_classes=num_classes,
                                               name='confusion_matrix')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        summaries.add(tf.summary.scalar('accuracy', accuracy))

        # Add summaries for model variables.
        for model_var in slim.get_model_variables():
            summaries.add(tf.summary.histogram(model_var.op.name, model_var))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        learning_rate = train_utils.get_model_learning_rate(
            FLAGS.learning_policy, FLAGS.base_learning_rate,
            FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
            FLAGS.training_number_of_steps, FLAGS.learning_power,
            FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        # total_loss, grads_and_vars = train_utils.optimize(optimizer, var_list=tf.trainable_variables())
        total_loss, grads_and_vars = train_utils.optimize(optimizer)
        total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan', name='total_loss')
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        grad_summ_op = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in grads_and_vars])

        # Create gradient update op.
        grad_updates = optimizer.apply_gradients(grads_and_vars,
                                                 global_step=global_step)

        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_op = tf.identity(total_loss, name='train_op')

        # Add the summaries. These contain the summaries created by model
        # and either optimize() or _gather_loss()
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries))
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir, graph)
        validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation', graph)

        #####################
        # prepare data
        #####################
        tfrecord_filenames = tf.placeholder(tf.string, shape=[])
        dataset = data.Dataset(tfrecord_filenames,
                               FLAGS.resize_height,
                               FLAGS.resize_width,
                               FLAGS.how_many_training_epochs,
                               batch_size=FLAGS.batch_size)
        iterator = dataset.dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            if FLAGS.saved_checkpoint_dir:
                if tf.gfile.IsDirectory(FLAGS.train_logdir):
                    checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_logdir)
                else:
                    checkpoint_path = FLAGS.train_logdir
                saver.restore(sess, checkpoint_path)

            # if FLAGS.pre_trained_checkpoint is not None
            if FLAGS.pre_trained_checkpoint:
                train_utils.restore_fn(FLAGS)

            start_epoch = 0
            train_batches = int(MINC2500_TRAIN_DATA_SIZE / FLAGS.batch_size)
            if MINC2500_TRAIN_DATA_SIZE % FLAGS.batch_size > 0:
                train_batches += 1
            validate_batches = int(MINC2500_VALIDATE_DATA_SIZE / FLAGS.batch_size)
            if MINC2500_VALIDATE_DATA_SIZE % FLAGS.batch_size > 0:
                validate_batches += 1

            # The filenames argument to the TFRecordDataset initializer can either
            # be a string, a list of strings, or a tf.Tensor of strings.
            train_record_filenames = os.path.join(FLAGS.dataset_dir, 'train.record')
            validate_record_filenames = os.path.join(FLAGS.dataset_dir, 'validate.record')
            ##################
            # training loop.
            ##################
            for num_epoch in range(start_epoch, FLAGS.how_many_training_epochs):
                tf.logging.info('--------------------------')
                tf.logging.info(' Epoch %d' % num_epoch)
                tf.logging.info('--------------------------')

                sess.run(iterator.initializer, feed_dict={tfrecord_filenames: train_record_filenames})
                for step in range(train_batches):
                    train_batch_xs, train_batch_ys = sess.run(next_batch)
                    # # TODO: make verify_image func.
                    # # # assert not np.any(np.isnan(train_batch_xs))
                    # n_batch = train_batch_xs.shape[0]
                    # for i in range(n_batch):
                    #     img = train_batch_xs[i]
                    #     # scipy.misc.toimage(img).show()
                    #     # Or
                    #     img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    #     cv2.imwrite('/home/ace19/Pictures/' + str(i) + '.png', img)
                    #     # cv2.imshow(str(train_batch_ys[idx]), img)
                    #     cv2.waitKey(100)
                    #     cv2.destroyAllWindows()

                    lr, train_summary, train_accuracy, train_loss, grad_vals, _ = \
                        sess.run([learning_rate, summary_op, accuracy, total_loss, grad_summ_op, train_op],
                                 feed_dict={X: train_batch_xs,
                                            ground_truth: train_batch_ys,
                                            is_training: True})

                    train_writer.add_summary(train_summary, num_epoch)
                    train_writer.add_summary(grad_vals, num_epoch)
                    tf.logging.info('Epoch #%d, Step #%d, rate %.10f, accuracy %.1f%%, loss %f' %
                                    (num_epoch, step, lr, train_accuracy * 100, train_loss))

                #################
                # validation
                #################
                tf.logging.info('--------------------------')
                tf.logging.info(' Start validation ')
                tf.logging.info('--------------------------')
                # Reinitialize iterator with the validation dataset
                sess.run(iterator.initializer, feed_dict={tfrecord_filenames: validate_record_filenames})
                total_val_accuracy = 0
                validation_count = 0
                total_conf_matrix = None
                for step in range(validate_batches):
                    validation_batch_xs, validation_batch_ys = sess.run(next_batch)
                    # Run a validation step and capture training summaries for TensorBoard
                    # with the `merged` op.
                    validation_summary, validation_accuracy, conf_matrix = sess.run(
                        [summary_op, accuracy, confusion_matrix],
                        feed_dict={
                            X: validation_batch_xs,
                            ground_truth: validation_batch_ys,
                            # learning_rate: FLAGS.base_learning_rate,
                            is_training: False
                        })

                    validation_writer.add_summary(validation_summary, num_epoch)

                    total_val_accuracy += validation_accuracy
                    validation_count += 1
                    if total_conf_matrix is None:
                        total_conf_matrix = conf_matrix
                    else:
                        total_conf_matrix += conf_matrix

                total_val_accuracy /= validation_count

                tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
                tf.logging.info('Validation accuracy = %.1f%% (N=%d)' %
                                (total_val_accuracy * 100, MINC2500_VALIDATE_DATA_SIZE))

                # Save the model checkpoint periodically.
                if (num_epoch <= FLAGS.how_many_training_epochs-1):
                    checkpoint_path = os.path.join(FLAGS.train_logdir, FLAGS.ckpt_name_to_save)
                    tf.logging.info('Saving to "%s-%d"', checkpoint_path, num_epoch)
                    saver.save(sess, checkpoint_path, global_step=num_epoch)


if __name__ == '__main__':
    tf.app.run()