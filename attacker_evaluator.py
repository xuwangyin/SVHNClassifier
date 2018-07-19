import tensorflow as tf
from donkey import Donkey
from model import Model, Reconstructor


class AttackerEvaluator(object):
    def __init__(self, path_to_eval_log_dir):
        self.summary_writer = tf.summary.FileWriter(path_to_eval_log_dir)

    def evaluate(self, path_to_restore_model_checkpoint_file, path_to_restore_attacker_checkpoint_file, path_to_tfrecords_file, num_examples, global_step):
        batch_size = 128
        num_batches = num_examples // batch_size
        needs_include_length = False

        with tf.Graph().as_default():
            image_batch, length_batch, digits_batch = Donkey.build_batch(path_to_tfrecords_file,
                                                                         num_examples=num_examples,
                                                                         batch_size=batch_size,
                                                                         shuffled=False)
            with tf.variable_scope('model'):
                length_logits, digits_logits, hidden_out = Model.inference(image_batch, drop_rate=0.0)
            with tf.variable_scope('attacker'):
                recovered = Reconstructor.recover_hidden(hidden_out)
                recovered = tf.image.resize_images(recovered, size=(54, 54))
            recovered_ssim = tf.reduce_mean(tf.image.ssim(tf.image.rgb_to_grayscale(image_batch), tf.image.rgb_to_grayscale(recovered),max_val=2))
            attacker_loss = -recovered_ssim
            length_predictions = tf.argmax(length_logits, axis=1)
            digits_predictions = tf.argmax(digits_logits, axis=2)

            if needs_include_length:
                labels = tf.concat([tf.reshape(length_batch, [-1, 1]), digits_batch], axis=1)
                predictions = tf.concat([tf.reshape(length_predictions, [-1, 1]), digits_predictions], axis=1)
            else:
                labels = digits_batch
                predictions = digits_predictions

            labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
            predictions_string = tf.reduce_join(tf.as_string(predictions), axis=1)

            accuracy, update_accuracy = tf.metrics.accuracy(
                labels=labels_string,
                predictions=predictions_string
            )

            tf.summary.image('image', image_batch)
            tf.summary.image('recovered_image', recovered)
            tf.summary.scalar('recovered_ssim', recovered_ssim)
            tf.summary.scalar('attacker_loss', attacker_loss)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.histogram('variables',
                                 tf.concat([tf.reshape(var, [-1]) for var in tf.trainable_variables()], axis=0))
            summary = tf.summary.merge_all()

            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                model_saver = tf.train.Saver(
                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model'))
                attacker_saver = tf.train.Saver(
                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attacker'))
                model_saver.restore(sess, path_to_restore_model_checkpoint_file)
                attacker_saver.restore(sess, path_to_restore_attacker_checkpoint_file)

                for _ in range(num_batches):
                    sess.run(update_accuracy)

                accuracy_val, summary_val = sess.run([accuracy, summary])
                self.summary_writer.add_summary(summary_val, global_step=global_step)

                coord.request_stop()
                coord.join(threads)

        return accuracy_val
