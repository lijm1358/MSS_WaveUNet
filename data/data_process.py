import os
from subprocess import call
import subprocess
import warnings

import numpy as np
import librosa 
from tqdm import tqdm


def separate_source(base_dir, target_dir):
    """Separate multiple input streams into 5 outputs.

    Original MUSDB18 data has 4 stems: vocals, drums, bass, and others.
    It separates those 4 stems and original sound. Each output has the number
    0 to 4, each indicates original sound, vocal stem, drum stem, bass stem, and others stem.

    Args:
        base_dir: Directory where the MUSDB18 dataset is stored
        target_dir: Directory where the separated stems will be stored
    """
    files = os.listdir(base_dir)
    for file in tqdm(files, position=0, leave=True):
        file_in = os.path.join(base_dir,file)
        for i in range(5):
            filesp = file.split(".")
            filesp[-1] = f"{i}.mp3"
            filesp = ".".join(filesp)
            file_out = os.path.join(f"{target_dir}", f"{filesp}")
            call(('ffmpeg', '-y', '-i', file_in, '-map', f'0:{i}', '-vn', file_out),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)

def convert_to_numpy(base_dir, target_dir):
    """Convert sound data into numpy array"""
    music_list = os.listdir(base_dir)
    music_list.sort()
    warnings.filterwarnings('ignore')
    for music in tqdm(music_list):
        music_path = os.path.join(base_dir, music)
        if not os.path.isdir(music_path):
            outfile_name = music
            outfile_name = os.path.join(target_dir, outfile_name)
            arr, _ = librosa.load(music_path)
            np.save(outfile_name, arr)

def separate_segment(base_dir, target_dir, sep_length):
    """Separate numpy array data into the same length segments

    Args:
        base_dir: Directory where the converted MUSDB18 dataset(numpy array) is stored
        target_dir: Directory where the array segments will be stored
        sep_length: the length of segments. set to 22050(Original sampling rate) for
                    1 second length segment.
    """
    filelist = os.listdir(base_dir)
    filelist.sort()
    song_np_full = []
    count = 0
    for i, filename in enumerate(tqdm(filelist)):
        song_np = np.load(os.path.join(base_dir, filename))
        song_np_full.append(song_np)
        if i%5==4:
            song_np_full = np.stack(song_np_full)
            index = 0
            while index+sep_length <= song_np_full.shape[1]:
                np.save(os.path.join(target_dir, str(count)), 
                        song_np_full[:, index:index+sep_length])
                index+=sep_length
                count+=1
            song_np_full = []
            
def data_augmentation(base_dir):
    def rename_augmented(filename, stem):
        file_split = filename.split('.')
        file_split[0] += "_aug"
        file_split[-3] = str(stem)
        file_split = '.'.join(file_split)
        return file_split
    
    filelist = os.listdir(base_dir)
    filelist.sort()
    song_np_list = []
    
    for i, filename in enumerate(tqdm(filelist)):
        filename_full = os.path.join(base_dir, filename)
        song_np = np.load(filename_full)
        song_np_list.append(song_np)
        if i%5 == 4:
            for n in range(1, 5):
                factor = np.random.uniform(0.7, 1.0)
                song_np_list[n] *= factor
            song_np_list[0] = sum(song_np_list[1:])
            for stem, song in enumerate(song_np_list):
                filename_renamed = rename_augmented(filename, stem)
                np.save(os.path.join(base_dir, filename_renamed), song)
            song_np_list.clear()