import tensorflow as tf
import PIL.Image
from math import pi

import argparse
import os
import io

import dataset_utils

parser = argparse.ArgumentParser(description='create tf record')
parser.add_argument('--set', default='Train', type=str, help='Convert training set, validation set or test set.')
parser.add_argument('--data_dir', default='/home/caijunhao/data/cmu_patch_datasets', type=str, help='Path of dataset.')
parser.add_argument('--output_path', default='', type=str, required=True, help='Path of record.')
parser.add_argument('--image_size', default=224, type=int, help='Image size.')
args = parser.parse_args()

sets = ['Train', 'Test', 'Validation']
labels = ['positive', 'negative']
label_file = 'dataInfo.txt'
folder = 'Images'


def convert_theta(theta):
    # theta : a num from -pi/2 to pi/2
    # return : a num from 0 to 17
    if theta > pi / 2:
        theta = theta - pi
    if theta < -pi / 2:
        theta = theta + pi
    theta = (theta + pi / 2) * 180 / pi  # [0,pi]
    diff = [abs(theta - i * 10) for i in xrange(18)]
    return diff.index(min(diff))


def dict_to_tf_example(path, size, label, theta):
    with tf.gfile.GFile(path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    class_label = 1 if label == 'positive' else 0
    theta_label = convert_theta(float(theta))
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': dataset_utils.bytes_feature(encoded_jpg),
        'image/format': dataset_utils.bytes_feature('jpeg'),
        'image/class/label': dataset_utils.int64_feature(class_label),
        'image/theta/label': dataset_utils.int64_feature(theta_label),
    }))
    return example


def main():
    if args.set not in sets:
        raise ValueError('set must be in : {}'.format(sets))

    data_dir = os.path.join(args.data_dir, args.set)
    for label in labels:
        writer = tf.python_io.TFRecordWriter(os.path.join(args.output_path, args.set+'_'+label+'.tfrecord'))
        data_info = dataset_utils.read_examples_list(os.path.join(data_dir, label, label_file))
        for image_name, theta in data_info:
            image_path = os.path.join(data_dir, label, folder, image_name)
            tf_example = dict_to_tf_example(image_path, args.image_size, label, theta)
            writer.write(tf_example.SerializeToString())
        writer.close()


if __name__ == '__main__':
    main()
