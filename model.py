import tensorflow as tf
import tensorflow.contrib.slim as slim


trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def model_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        biases_initializer=tf.constant_initializer(0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                return arg_sc


def model(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          reuse=False,
          spatial_squeeze=True,
          scope='alexnet_v2',
          adapt_scope=None,
          adapt_dims=128):
    # default image size = 224
    with tf.variable_scope(scope, 'alexnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                              scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                                  scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout7')
                if adapt_scope is not None:
                    net = slim.conv2d(net, adapt_dims, [1, 1], scope=adapt_scope)
                net = slim.conv2d(net, num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=tf.zeros_initializer(),
                                  scope='fc8')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
    return net, end_points


def se_model(inputs,
             num_classes=1000,
             is_training=True,
             dropout_keep_prob=0.5,
             reuse=False,
             spatial_squeeze=True,
             scope='alexnet_v2',
             adapt_scope=None,
             adapt_dims=128,
             reduction_ratio=16):
    # default image size = 224
    with tf.variable_scope(scope, 'alexnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                              scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                                  scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout7')
                if adapt_scope is not None:
                    net = slim.conv2d(net, adapt_dims, [1, 1], scope=adapt_scope)
                    channel_size = net.get_shape().as_list()[-1]
                    sqeeze = slim.conv2d(net, channel_size / reduction_ratio, [1, 1], scope=adapt_scope+'/squeeze')
                    excitation = slim.conv2d(sqeeze, channel_size, [1, 1], activation_fn=None,
                                             scope=adapt_scope+'/excitation')
                    sigmoid_activation = tf.nn.sigmoid(excitation, name='sigmoid_activation')
                    net = tf.multiply(net, sigmoid_activation)
                net = slim.conv2d(net, num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=tf.zeros_initializer(),
                                  scope='fc8')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            end_points['sigmoid_activation'] = sigmoid_activation
            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
    return net, end_points


def full_senet(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               reuse=False,
               spatial_squeeze=True,
               scope='alexnet_v2',
               adapt_scope=None,
               adapt_dims=128,
               reduction_ratio=16):
    # default image size = 224
    with tf.variable_scope(scope, 'alexnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                              scope='conv1')
            net = se_module(net, reduction_ratio=reduction_ratio, scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')
            net = se_module(net, reduction_ratio=reduction_ratio, scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = se_module(net, reduction_ratio=reduction_ratio, scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = se_module(net, reduction_ratio=reduction_ratio, scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = se_module(net, reduction_ratio=reduction_ratio, scope='conv5')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                                  scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout7')
                if adapt_scope is not None:
                    net = slim.conv2d(net, adapt_dims, [1, 1], scope=adapt_scope)
                    net = se_module(net,reduction_ratio=reduction_ratio, scope=adapt_scope)
                net = slim.conv2d(net, num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=tf.zeros_initializer(),
                                  scope='fc8')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
    return net, end_points


def se_module(inputs, reduction_ratio=16, scope=''):
    net = tf.reduce_mean(inputs, [1, 2], name=scope+'/global_pooling', keepdims=True)
    channel_size = net.get_shape().as_list()[-1]
    sqeeze = slim.conv2d(net, channel_size / reduction_ratio, [1, 1], scope=scope + '/squeeze')
    excitation = slim.conv2d(sqeeze, channel_size, [1, 1], activation_fn=None, scope=scope + '/excitation')
    sigmoid_activation = tf.nn.sigmoid(excitation, name=scope+'/sigmoid_activation')
    net = tf.multiply(inputs, sigmoid_activation)
    return net




