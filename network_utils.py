import tensorflow as tf
import tensorflow.contrib.slim as slim

import os


def get_dataset(dataset_dir, num_readers, num_preprocessing_threads, hparams, reader=None):
    dataset_dir_list = [os.path.join(dataset_dir, filename)
                        for filename in os.listdir(dataset_dir) if filename.endswith('.tfrecord')]
    if reader is None:
        reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label': tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
        'image/theta/label': tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(channels=3),
        'class_label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
        'theta_label': slim.tfexample_decoder.Tensor('image/theta/label', shape=[]),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    dataset = slim.dataset.Dataset(data_sources=dataset_dir_list,
                                   reader=reader,
                                   decoder=decoder,
                                   num_samples=288918,
                                   items_to_descriptions=None)
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              num_readers=num_readers,
                                                              common_queue_capacity=20 * hparams.batch_size,
                                                              common_queue_min=10 * hparams.batch_size)
    [image, class_label, theta_label] = provider.get(['image', 'class_label', 'theta_label'])
    image = tf.image.resize_images(image, [hparams.image_size, hparams.image_size])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image -= [123.68, 116.779, 103.939]
    images, class_labels, theta_labels = tf.train.batch([image, class_label, theta_label],
                                                        batch_size=hparams.batch_size,
                                                        num_threads=num_preprocessing_threads,
                                                        capacity=5*hparams.batch_size)
    return images, class_labels, theta_labels


def create_loss(scores, class_labels, theta_labels):
    theta_lebels_one_hot = tf.one_hot(theta_labels, depth=18, on_value=1.0, off_value=0.0)
    theta_acted = tf.reduce_sum(tf.multiply(scores, theta_lebels_one_hot), axis=1, name='theta_acted')
    sig_op = tf.clip_by_value(slim.nn.sigmoid(theta_acted), 0.001, 0.999, name='clipped_sigmoid')
    sig_loss = - tf.to_float(class_labels) * tf.log(sig_op) - \
               (1 - tf.to_float(class_labels)) * tf.log(1 - sig_op)
    loss = tf.reduce_mean(sig_loss)
    conf = tf.equal(tf.to_int32(tf.greater_equal(sig_op, 0.5)),
                    tf.to_int32(tf.greater_equal(tf.to_float(class_labels), 0.1)))
    accuracy = tf.reduce_mean(tf.to_float(conf))
    return loss, accuracy


def add_summary(images, end_points, loss, accuracy, scope='graspnet'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('image', images[0:1, :, :, :])
    for i in range(64):
        # tf.summary.image(scope+'/conv1/conv1_1'+'_{}'.format(i), end_points[scope+'/conv1/conv1_1'][0:1, :, :, i:i+1])
        tf.summary.image(scope + '/conv1' + '_{}'.format(i),
                         end_points[scope + '/conv1'][0:1, :, :, i:i + 1])
    variable_list = slim.get_model_variables(scope=scope)
    for var in variable_list:
        tf.summary.histogram(var.name, var)


def restore_map(from_graspnet_checkpoint, scope, model_name, checkpoint_exclude_scope=''):
    if not from_graspnet_checkpoint:
        variables_to_restore = restore_from_classification_checkpoint(scope, model_name, checkpoint_exclude_scope)
        return variables_to_restore
    else:
        variable_list = slim.get_model_variables(scope)
        variables_to_restore = {var.op.name: var for var in variable_list}
        return variables_to_restore


def restore_from_classification_checkpoint(scope, model_name, checkpoint_exclude_scope):
    variable_list = slim.get_model_variables(scope)
    variable_list = [var for var in variable_list if checkpoint_exclude_scope not in var.op.name]
    variables_to_restore = {}
    for var in variable_list:
        if var.name.startswith(scope):
            var_name = var.op.name.replace(scope, model_name)
            variables_to_restore[var_name] = var
    return variables_to_restore

