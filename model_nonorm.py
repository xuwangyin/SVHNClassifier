import tensorflow as tf

class Attacker(object):
    @staticmethod
    def recover_hidden(attacker_type, hidden_out, is_training, defend_layer='hidden4'):
        assert attacker_type in ['dense', 'deconv']
        if attacker_type == 'dense':
            return DenseAttacker.recover_hidden(hidden_out, is_training, defend_layer)
        if attacker_type == 'deconv':
            return DeconvAttacker.recover_hidden(hidden_out, is_training, defend_layer)

class DenseAttacker(object):
    @staticmethod
    def recover_hidden(hidden_out, is_training, defend_layer='hidden4'):
        with tf.variable_scope('dense_reconstructor'):
            flatten = tf.reshape(hidden_out, [32, -1])
            dense = tf.layers.dense(flatten, units=1024, activation=tf.nn.relu)
            dense = tf.layers.dense(dense, units=54*54*3, activation=tf.nn.relu)
            image = tf.reshape(dense, [-1, 54, 54, 3])
            return image


class DeconvAttacker(object):
    @staticmethod
    def recover_hidden(hidden_out, is_training, defend_layer='hidden4'):
        assert defend_layer in ['hidden4', 'hidden6', 'hidden8']
        if defend_layer == 'hidden4':
            return DeconvAttacker.recover_hidden4(hidden_out, is_training)
        if defend_layer == 'hidden6':
            return DeconvAttacker.recover_hidden6(hidden_out, is_training)
        if defend_layer == 'hidden8':
            return DeconvAttacker.recover_hidden8(hidden_out, is_training)

    @staticmethod
    def recover_hidden4(hidden_out, is_training):
        with tf.variable_scope('recover_hidden4'):
            # recover from hidden 4
            r_pool = tf.layers.Conv2DTranspose(filters=160, kernel_size=(2, 2), strides=(1, 1), padding='same')(hidden_out)
            r_relu = tf.nn.relu(r_pool)
            r_norm = tf.layers.batch_normalization(r_relu)
            r_conv = tf.layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), padding='same')(r_norm)

            r_pool = tf.layers.Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same')(r_conv)
            r_relu = tf.nn.relu(r_pool)
            r_norm = tf.layers.batch_normalization(r_relu)
            r_conv = tf.layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), padding='same')(r_norm)

            r_pool = tf.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(1, 1), padding='same')(r_conv)
            r_relu = tf.nn.relu(r_pool)
            r_norm = tf.layers.batch_normalization(r_relu)
            r_conv = tf.layers.Conv2DTranspose(filters=48, kernel_size=(5, 5), padding='same')(r_norm)

            r_pool = tf.layers.Conv2DTranspose(filters=48, kernel_size=(2, 2), strides=(2, 2), padding='same')(r_conv)
            r_relu = tf.nn.relu(r_pool)
            r_norm = tf.layers.batch_normalization(r_relu)
            r_conv = tf.layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), padding='same')(r_norm)
            recovered = tf.image.resize_images(r_conv, size=(54, 54))
            return recovered

    @staticmethod
    def recover_hidden6(hidden_out, is_training):
        with tf.variable_scope('recover_hidden6'):
            # reverse hidden6
            r_pool = tf.layers.Conv2DTranspose(filters=192, kernel_size=(2, 2), strides=(1, 1), padding='same')(hidden_out)
            r_relu = tf.nn.relu(r_pool)
            r_norm = tf.layers.batch_normalization(r_relu)
            r_conv = tf.layers.Conv2DTranspose(filters=192, kernel_size=(5, 5), padding='same')(r_norm)

            # reverse hidden5
            r_pool = tf.layers.Conv2DTranspose(filters=192, kernel_size=(2, 2), strides=(2, 2), padding='same')(r_conv)
            r_relu = tf.nn.relu(r_pool)
            r_norm = tf.layers.batch_normalization(r_relu)
            r_conv = tf.layers.Conv2DTranspose(filters=160, kernel_size=(5, 5), padding='same')(r_norm)

            return DeconvAttacker.recover_hidden4(r_conv, is_training)

    @staticmethod
    def recover_hidden8(hidden_out, is_training):
        with tf.variable_scope('recover_hidden8'):
            # # hidden 10 reverse
            # r_dense = tf.layers.dense(hidden_out, units=3072, activation=tf.nn.relu)
            # # hidden 9 reverse
            # r_dense = tf.layers.dense(r_dense, units=4*4*192, activation=tf.nn.relu)
            #
            # r_flatten = tf.reshape(r_dense, [-1, 4, 4, 192])

            # reverse hidden8
            r_pool = tf.layers.Conv2DTranspose(filters=192, kernel_size=(2, 2), strides=(1, 1), padding='same')(hidden_out)
            r_relu = tf.nn.relu(r_pool)
            r_norm = tf.layers.batch_normalization(r_relu)
            r_conv = tf.layers.Conv2DTranspose(filters=192, kernel_size=(5, 5), padding='same')(r_norm)

            # reverse hidden7
            r_pool = tf.layers.Conv2DTranspose(filters=192, kernel_size=(2, 2), strides=(2, 2), padding='same')(r_conv)
            r_relu = tf.nn.relu(r_pool)
            r_norm = tf.layers.batch_normalization(r_relu)
            r_conv = tf.layers.Conv2DTranspose(filters=192, kernel_size=(5, 5), padding='same')(r_norm)

            return DeconvAttacker.recover_hidden6(r_conv, is_training)

class Model(object):

    @staticmethod
    def inference(x, drop_rate, is_training, defend_layer='hidden4'):
        assert defend_layer in ['hidden4', 'hidden6', 'hidden8']
        with tf.variable_scope('hidden1'):
            conv = tf.layers.conv2d(x, filters=48, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden1 = dropout

        with tf.variable_scope('hidden2'):
            conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden2 = dropout

        with tf.variable_scope('hidden3'):
            conv = tf.layers.conv2d(hidden2, filters=128, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden3 = dropout

        with tf.variable_scope('hidden4'):
            conv = tf.layers.conv2d(hidden3, filters=160, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden4 = dropout

        with tf.variable_scope('hidden5'):
            conv = tf.layers.conv2d(hidden4, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden5 = dropout

        with tf.variable_scope('hidden6'):
            conv = tf.layers.conv2d(hidden5, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden6 = dropout

        with tf.variable_scope('hidden7'):
            conv = tf.layers.conv2d(hidden6, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden7 = dropout

        with tf.variable_scope('hidden8'):
            conv = tf.layers.conv2d(hidden7, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden8 = dropout

        flatten = tf.reshape(hidden8, [-1, 4 * 4 * 192])

        with tf.variable_scope('hidden9'):
            dense = tf.layers.dense(flatten, units=3072, activation=tf.nn.relu)
            hidden9 = dense

        with tf.variable_scope('hidden10'):
            dense = tf.layers.dense(hidden9, units=3072, activation=tf.nn.relu)
            hidden10 = dense

        with tf.variable_scope('digit_length'):
            dense = tf.layers.dense(hidden10, units=7)
            length = dense

        with tf.variable_scope('digit1'):
            dense = tf.layers.dense(hidden10, units=11)
            digit1 = dense

        with tf.variable_scope('digit2'):
            dense = tf.layers.dense(hidden10, units=11)
            digit2 = dense

        with tf.variable_scope('digit3'):
            dense = tf.layers.dense(hidden10, units=11)
            digit3 = dense

        with tf.variable_scope('digit4'):
            dense = tf.layers.dense(hidden10, units=11)
            digit4 = dense

        with tf.variable_scope('digit5'):
            dense = tf.layers.dense(hidden10, units=11)
            digit5 = dense

        length_logits, digits_logits = length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)
        if defend_layer == 'hidden4':
            return length_logits, digits_logits, hidden4
        if defend_layer == 'hidden6':
            return length_logits, digits_logits, hidden6
        if defend_layer == 'hidden8':
            return length_logits, digits_logits, hidden8

    @staticmethod
    def loss(length_logits, digits_logits, length_labels, digits_labels):
        length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=length_logits))
        digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0], logits=digits_logits[:, 0, :]))
        digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
        digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
        digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
        digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4], logits=digits_logits[:, 4, :]))
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
        return loss
