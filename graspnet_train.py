from hparams import create_params
from network_utils import get_dataset, create_loss, add_summary, restore_map
from nets import vgg, alexnet

import tensorflow as tf
import tensorflow.contrib.slim as slim

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser(description='training network')

parser.add_argument('--master', default='', type=str, help='BNS name of the TensorFlow master to use')
parser.add_argument('--task_id', default=0, type=int, help='The Task ID. This value is used '
                                                           'when training with multiple workers to '
                                                           'identify each worker.')
parser.add_argument('--train_log_dir', default='logs/', type=str, help='Directory where to write event logs.')
parser.add_argument('--save_summaries_steps', default=120, type=int, help='The frequency with which'
                                                                          ' summaries are saved, in seconds.')
parser.add_argument('--save_interval_secs', default=600, type=int, help='The frequency with which '
                                                                        'the model is saved, in seconds.')
parser.add_argument('--print_loss_steps', default=100, type=int, help='The frequency with which '
                                                                      'the losses are printed, in steps.')
parser.add_argument('--dataset_dir', default='', type=str, help='The directory where the datasets can be found.')
parser.add_argument('--num_readers', default=4, type=int, help='The number of parallel readers '
                                                               'that read data from the dataset.')
parser.add_argument('--num_steps', default=20000, type=int, help='The max number of gradient steps to take '
                                                                 'during training.')
parser.add_argument('--num_preprocessing_threads', default=4, type=int, help='The number of threads '
                                                                             'used to create the batches.')
parser.add_argument('--hparams', default='', type=str, help='Comma separated hyper parameter values')
parser.add_argument('--from_graspnet_checkpoint', default=False, type=bool, help='Whether load checkpoint '
                                                                                 'from graspnet checkpoint '
                                                                                 'or classification checkpoint.')
parser.add_argument('--checkpoint_dir', default='', type=str, help='The directory where the checkpoint can be found')
args = parser.parse_args()
num_classes = 18


def train(run_dir,
          master,
          task_id,
          num_readers,
          from_graspnet_checkpoint,
          dataset_dir,
          checkpoint_dir,
          save_summaries_steps,
          save_interval_secs,
          num_preprocessing_threads,
          num_steps,
          hparams,
          scope='graspnet'):
    for path in [run_dir]:
        if not tf.gfile.Exists(path):
            tf.gfile.Makedirs(path)
    hparams_filename = os.path.join(run_dir, 'hparams.json')
    with tf.gfile.FastGFile(hparams_filename, 'w') as f:
        f.write(hparams.to_json())
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(task_id)):
            global_step = slim.get_or_create_global_step()
            images, class_labels, theta_labels = get_dataset(dataset_dir,
                                                             num_readers,
                                                             num_preprocessing_threads,
                                                             hparams)
            '''
            with slim.arg_scope(vgg.vgg_arg_scope()):
                net, end_points = vgg.vgg_16(inputs=images,
                                             num_classes=num_classes,
                                             is_training=True,
                                             dropout_keep_prob=0.7,
                                             scope=scope)
            '''
            with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
                net, end_points = alexnet.alexnet_v2(inputs=images,
                                                     num_classes=num_classes,
                                                     is_training=True,
                                                     dropout_keep_prob=0.7,
                                                     scope=scope)
            loss, accuracy = create_loss(net, class_labels, theta_labels)
            learning_rate = hparams.learning_rate
            if hparams.lr_decay_step:
                learning_rate = tf.train.exponential_decay(hparams.learning_rate,
                                                           slim.get_or_create_global_step(),
                                                           decay_steps=hparams.lr_decay_step,
                                                           decay_rate=hparams.lr_decay_rate,
                                                           staircase=True)
            tf.summary.scalar('Learning_rate', learning_rate)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = slim.learning.create_train_op(loss, optimizer)
            add_summary(images, end_points, loss, accuracy, scope=scope)
            summary_op = tf.summary.merge_all()
            variable_map = restore_map(from_graspnet_checkpoint=from_graspnet_checkpoint,
                                       scope=scope,
                                       model_name=hparams.model_name,
                                       checkpoint_exclude_scope='fc8')
            init_saver = tf.train.Saver(variable_map)

            def initializer_fn(sess):
                init_saver.restore(sess, checkpoint_dir)
                tf.logging.info('Successfully load pretrained checkpoint.')

            init_fn = initializer_fn
            session_config = tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False)
            session_config.gpu_options.allow_growth = True
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=save_interval_secs,
                                   max_to_keep=100)

            slim.learning.train(train_op,
                                logdir=run_dir,
                                master=master,
                                global_step=global_step,
                                session_config=session_config,
                                # init_fn=init_fn,
                                summary_op=summary_op,
                                number_of_steps=num_steps,
                                startup_delay_steps=15,
                                save_summaries_secs=save_summaries_steps,
                                saver=saver)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = create_params(args.hparams)
    train(run_dir=args.train_log_dir,
          master=args.master,
          task_id=args.task_id,
          num_readers=args.num_readers,
          from_graspnet_checkpoint=args.from_graspnet_checkpoint,
          dataset_dir=args.dataset_dir,
          checkpoint_dir=args.checkpoint_dir,
          save_summaries_steps=args.save_summaries_steps,
          save_interval_secs=args.save_interval_secs,
          num_preprocessing_threads=args.num_preprocessing_threads,
          num_steps=args.num_steps,
          hparams=hparams,
          scope='graspnet')


if __name__ == '__main__':
    main()
