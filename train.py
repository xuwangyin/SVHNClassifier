import os
from datetime import datetime
import time
import tensorflow as tf
from meta import Meta
from donkey import Donkey
# from model import Model, Reconstructor
from model_nonorm import Model, Attacker
from evaluator import Evaluator

tf.app.flags.DEFINE_string('data_dir', './data', 'Directory to read TFRecords files')
tf.app.flags.DEFINE_string('train_logdir', './logs/train', 'Directory to write training logs')
tf.app.flags.DEFINE_string('restore_checkpoint', None,
                           'Path to restore checkpoint (without postfix), e.g. ./logs/train/model.ckpt-100')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Default 32')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Default 1e-2')
tf.app.flags.DEFINE_integer('patience', 100, 'Default 100, set -1 to train infinitely')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'Default 10000')
tf.app.flags.DEFINE_float('decay_rate', 0.9, 'Default 0.9')
tf.app.flags.DEFINE_float('ssim_weight', 1.0, 'Default 1.0')
tf.app.flags.DEFINE_string('defend_layer', 'hidden4', 'Default hidden4')
tf.app.flags.DEFINE_string('attacker_type', 'deconv', 'deconv')
FLAGS = tf.app.flags.FLAGS


def _train(path_to_train_tfrecords_file, num_train_examples, path_to_val_tfrecords_file, num_val_examples,
           path_to_train_log_dir, path_to_restore_checkpoint_file, training_options):
    batch_size = training_options['batch_size']
    initial_patience = training_options['patience']
    num_steps_to_show_loss = 100
    num_steps_to_check = 1000

    with tf.Graph().as_default():
        image_batch, length_batch, digits_batch = Donkey.build_batch(path_to_train_tfrecords_file,
                                                                     num_examples=num_train_examples,
                                                                     batch_size=batch_size,
                                                                     shuffled=True)
        with tf.variable_scope('model'):
            length_logtis, digits_logits, hidden_out = Model.inference(image_batch, drop_rate=0.2, is_training=True, defend_layer=FLAGS.defend_layer)
        with tf.variable_scope('defender'):
            recovered = Attacker.recover_hidden(FLAGS.attacker_type, hidden_out, True, FLAGS.defend_layer)
        ssim = tf.reduce_mean(tf.abs(tf.image.ssim(image_batch, recovered, max_val=2)))
        model_loss = Model.loss(length_logtis, digits_logits, length_batch, digits_batch)
        loss = model_loss + FLAGS.ssim_weight * ssim
        defender_loss = -ssim

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(training_options['learning_rate'], global_step=global_step,
                                                   decay_steps=training_options['decay_steps'], decay_rate=training_options['decay_rate'], staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        model_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='model')
        with tf.control_dependencies(model_update_ops):
            train_op = optimizer.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='model'), global_step=global_step)

        defender_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='defender')
        with tf.control_dependencies(defender_update_ops):
            defender_op = optimizer.minimize(defender_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='defender'), global_step=global_step)

        tf.summary.image('image', image_batch, max_outputs=20)
        tf.summary.image('recovered', recovered, max_outputs=20)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('ssim', ssim)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(path_to_train_log_dir, sess.graph)
            evaluator = Evaluator(os.path.join(path_to_train_log_dir, 'eval/val'))

            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver = tf.train.Saver()
            model_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='model'))
            defender_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='defender'))
            # if path_to_restore_checkpoint_file is not None:
            #     assert tf.train.checkpoint_exists(path_to_restore_checkpoint_file), \
            #         '%s not found' % path_to_restore_checkpoint_file
            #     saver.restore(sess, path_to_restore_checkpoint_file)
            #     print('Model restored from file: %s' % path_to_restore_checkpoint_file)

            print('Start training')
            patience = initial_patience
            best_accuracy = 0.0
            duration = 0.0

            while True:
                start_time = time.time()
                _, _, loss_val, summary_val, global_step_val, learning_rate_val = sess.run([train_op, defender_op, loss, summary, global_step, learning_rate])
                duration += time.time() - start_time

                # print("image: {} - {}".format(image_batch_val.min(), image_batch_val.max()))

                if global_step_val % num_steps_to_show_loss == 0:
                    examples_per_sec = batch_size * num_steps_to_show_loss / duration
                    duration = 0.0
                    print('=> %s: step %d, loss = %f (%.1f examples/sec)' % (
                        datetime.now(), global_step_val, loss_val, examples_per_sec))

                if global_step_val % num_steps_to_check != 0:
                    continue

                summary_writer.add_summary(summary_val, global_step=global_step_val)

                print('=> Evaluating on validation dataset...')
                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir,'model_defender.ckpt'))
                model_saver.save(sess, os.path.join(path_to_train_log_dir, 'model.ckpt'))
                defender_saver.save(sess, os.path.join(path_to_train_log_dir, 'defender.ckpt'))
                accuracy = evaluator.evaluate(path_to_latest_checkpoint_file, path_to_val_tfrecords_file,
                                              num_val_examples,
                                              global_step_val,
                                              FLAGS.defend_layer,
                                              FLAGS.attacker_type)
                print('==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))

                if accuracy > best_accuracy:
                    model_saver.save(sess, os.path.join(path_to_train_log_dir, 'model_best.ckpt'))
                    defender_saver.save(sess, os.path.join(path_to_train_log_dir, 'defender_best.ckpt'))
                    patience = initial_patience
                    best_accuracy = accuracy
                else:
                    patience -= 1

                print('=> patience = %d' % patience)
                # if patience == 0:
                #     break

            coord.request_stop()
            coord.join(threads)
            print('Finished')


def main(_):
    path_to_train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    path_to_val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    path_to_tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')
    path_to_train_log_dir = FLAGS.train_logdir
    path_to_restore_checkpoint_file = FLAGS.restore_checkpoint
    training_options = {
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'patience': FLAGS.patience,
        'decay_steps': FLAGS.decay_steps,
        'decay_rate': FLAGS.decay_rate
    }

    meta = Meta()
    meta.load(path_to_tfrecords_meta_file)

    _train(path_to_train_tfrecords_file, meta.num_train_examples,
           path_to_val_tfrecords_file, meta.num_val_examples,
           path_to_train_log_dir, path_to_restore_checkpoint_file,
           training_options)


if __name__ == '__main__':
    tf.app.run(main=main)
