import tensorflow as tf


def create_params(hparam_string=None):
    hparams = tf.contrib.training.HParams(learning_rate=0.01,
                                          lr_decay_step=5000,
                                          lr_decay_rate=0.95,
                                          batch_size=64,
                                          image_size=224,
                                          model_name='alexnet_v2')

    if hparam_string:
        tf.logging.info('Parsing command line hparams: %s', hparam_string)
        hparams.parse(hparam_string)

    tf.logging.info('Final parsed hparams: %s', hparams.values())
    return hparams
