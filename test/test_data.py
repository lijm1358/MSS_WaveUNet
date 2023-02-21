import pytest
import os
from subprocess import call
import shutil

import numpy as np
import pydub

from data.data_process import *
from data.dataset import MUSDBDataset

def _remove_data_test():
    try:
        shutil.rmtree("datas")
    except FileNotFoundError:
        pass

def _create_audioset():
    os.makedirs("datas", exist_ok=True)
    os.makedirs(os.path.normpath("datas/train"), exist_ok=True)
    stems = []
    file_list = []
    for i in range(5):
        stems.append(np.random.rand(1000,))
    for i, stem in enumerate(stems):
        song = pydub.AudioSegment(stem.tobytes(), frame_rate=22050, sample_width=1, channels=1)
        filename = os.path.join("datas", f"out{i}.mp3")
        file_list.append(filename)
        song.export(filename)
    call(('ffmpeg', '-i', file_list[0], '-i', file_list[1], 
          '-i', file_list[2], '-i', file_list[3], '-i', file_list[4], '-c', 'copy', 
          '-map', '0:a', '-map', '1:a', '-map', '2:a', '-map', '3:a', '-map', '4:a',
          'datas/train/combined.mp4'),
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL)

def _create_numpy_dataset():
    np_path = os.path.normpath("datas/train_numpy")
    os.makedirs(np_path, exist_ok=True)
    for i in range(5):
        song = np.random.rand(1000,)
        np.save(os.path.join(np_path, f"out{i}.mp3.npy"), song)

def test_source_separate():
    _create_audioset()
    base_dir = os.path.join("datas", "train")
    target_dir = os.path.join("datas", "train_out")
    os.makedirs(target_dir, exist_ok=True)
    separate_source(base_dir, target_dir)

    assert len(os.listdir(target_dir)) == 5

def test_convert_numpy():
    _remove_data_test()
    _create_audioset()
    base_dir = "datas"
    target_dir = os.path.join("datas", "train_numpy")
    os.makedirs(target_dir, exist_ok=True)
    convert_to_numpy(base_dir, target_dir)

    filenames = os.listdir(target_dir)
    song_np = np.load(os.path.join(target_dir, filenames[0]))
    assert len(filenames) == 5
    assert all(file[-3:] == 'npy' for file in filenames)

@pytest.mark.parametrize("sep_length", [500, 800])
def test_segment_separate(sep_length):
    _remove_data_test()
    _create_numpy_dataset()
    base_dir = os.path.join("datas", "train_numpy")
    target_dir = os.path.join("datas", "train_split")
    os.makedirs(target_dir, exist_ok=True)
    separate_segment(base_dir, target_dir, sep_length)

    np_seg_list = os.listdir(target_dir)
    np_seg = np.load(os.path.join(target_dir, np_seg_list[0]))

    assert np_seg.shape == (5,sep_length)
    assert len(np_seg_list) == 1000//sep_length

def test_musdb18():
    _remove_data_test()

    data_dir = os.path.join("datas", "train_split")
    os.makedirs(data_dir, exist_ok=True)

    for i in range(10):
        np_arr = np.random.rand(5, 1000)
        np.save(os.path.join(data_dir, f"{i}.npy"), np_arr)

    ds = MUSDBDataset(data_dir)

    assert len(ds) == 10
    assert ds[0][0].shape == (1, 1000)
    assert ds[0][1].shape == (4, 1000)