import argparse
import os

parser = argparse.ArgumentParser(description='evaluate grasping accuracy.')
parser.add_argument('--file_path',
                    default='/home/caijunhao/ros_ws/src/graspnet/target_only_3088_result/grasping_label.txt',
                    type=str,
                    help='path to the file with grasping label.')
args = parser.parse_args()


def main():
    with open(args.file_path, 'r') as f:
        annotations = f.readlines()
    num_sample = len(annotations)
    num_success = 0
    for annotation in annotations:
        _, _, label = annotation.split(' ')
        num_success += 1 - int(label)
    print 'num_success: {}'.format(num_success)
    print 'num_trials: {}'.format(num_sample)
    print 'accuracy: {}'.format(num_success * 1.0 / num_sample)


if __name__ == '__main__':
    main()

