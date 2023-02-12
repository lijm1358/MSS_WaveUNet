import argparse
from argparse import RawTextHelpFormatter
import os

from data_process import separate_source, separate_segment, convert_to_numpy

def main(args):
    base_dir = args.base_dir
    target_dir = args.target_dir
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
        print("Separating stems")
        separate_source(base_dir_train, target_dir_train)
        separate_source(base_dir_test, target_dir_test)
    else:
        target_dir_train = base_dir_train
        target_dir_test = base_dir_test

    os.makedirs(target_dir_numpy_train, exist_ok=True)
    os.makedirs(target_dir_numpy_test, exist_ok=True)
    os.makedirs(target_dir_split_train, exist_ok=True)
    os.makedirs(target_dir_split_test, exist_ok=True)

    print("Converting to numpy array")
    convert_to_numpy(target_dir_train, target_dir_numpy_train)
    convert_to_numpy(target_dir_test, target_dir_numpy_test)

    print("Separating into segments")
    separate_segment(target_dir_numpy_train, target_dir_split_train, sep_length_train)
    separate_segment(target_dir_numpy_test, target_dir_split_test, sep_length_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process MUSDB18(https://sigsep.github.io/datasets/musdb.html) audio dataset into same length numpy array segments.',
                                     epilog='If you want to use your dataset, you may use your own data processing code.',
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('base_dir', type=str, help='The directory where MUSDB18 dataset is stored. Should contain the directory "train" and "test".')
    parser.add_argument('target_dir', type=str, help='The directory where the processed dataset will be stored')
    parser.add_argument('--sep_length_train', type=int, default=16384, help='length of the separated segments of train dataset. Default value is 16384 along with the baseline model of original paper.')
    parser.add_argument('--sep_length_test', type=int, default=22050, help='length of the separated segments of test dataset')
    parser.add_argument('--no_sep_stem', default=False, action='store_true', help='ignore the stream separation step(Optional)')

    args = parser.parse_args()

    main(args)