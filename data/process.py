import argparse
from argparse import RawTextHelpFormatter
import os

from data_process import separate_source, separate_segment, convert_to_numpy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process MUSDB18(https://sigsep.github.io/datasets/musdb.html) \
audio dataset into same length numpy array segments.',
                                     epilog='If you want to use custom dataset, you may use your own \
data processing code.',
                                    formatter_class=RawTextHelpFormatter)

    parser.add_argument('BASE_DIR', type=str, help='the directory where MUSDB18 dataset is stored')
    parser.add_argument('TARGET_DIR', type=str, help='the directory where the processed dataset will be stored')
    parser.add_argument('--sep_length_train', dest='sep_length_train', type=int, default=16384, help='length of the separated segments of train dataset')
    parser.add_argument('--sep_length_test', dest='sep_length_test', type=int, default=22050, help='length of the separated segments of test dataset')
    parser.add_argument('--no_sep_stem', default=False, action='store_true')

    args = parser.parse_args()

    base_dir = args.BASE_DIR
    target_dir = args.TARGET_DIR
    sep_length_train = args.sep_length_train
    sep_length_test = args.sep_length_test
    no_sep_stem = args.no_sep_stem

    base_dir_train = os.path.join(base_dir, 'train')
    base_dir_test = os.path.join(base_dir, 'test')

    target_dir_train = os.path.join(target_dir, 'train')
    target_dir_test = os.path.join(target_dir, 'test')

    target_dir_numpy_train = os.path.join(target_dir_train, 'data_numpy')
    target_dir_numpy_test = os.path.join(target_dir_test, 'data_numpy')

    target_dir_split_train = os.path.join(target_dir_train, 'data_split')
    target_dir_split_test = os.path.join(target_dir_test, 'data_split')

    os.makedirs(target_dir_train, exist_ok=True)
    os.makedirs(target_dir_test, exist_ok=True)

    if not no_sep_stem:
        separate_source(base_dir_train, target_dir_train)
        separate_source(base_dir_test, target_dir_test)
    else:
        target_dir_train = base_dir_train
        target_dir_test = base_dir_test

    os.makedirs(target_dir_numpy_train, exist_ok=True)
    os.makedirs(target_dir_numpy_test, exist_ok=True)
    os.makedirs(target_dir_split_train, exist_ok=True)
    os.makedirs(target_dir_split_test, exist_ok=True)

    convert_to_numpy(target_dir_train, target_dir_numpy_train)
    convert_to_numpy(target_dir_test, target_dir_numpy_test)

    separate_segment(target_dir_numpy_train, target_dir_split_train, sep_length_train)
    separate_segment(target_dir_numpy_test, target_dir_split_test, sep_length_test)