import os
from subprocess import call
import subprocess
import warnings

import numpy as np
import librosa 
from tqdm import tqdm


def separate_source(base_dir, target_dir):
    files = os.listdir(base_dir)
    for file in tqdm(files, position=0, leave=True):
        file_in = os.path.join(base_dir,file)
        for i in range(5):
            filesp = file.split(".")
            filesp[-1] = f"{i}.mp3"
            filesp = ".".join(filesp)
            file_out = f"{target_dir}/{filesp}"
            print(file_out)
            call(('ffmpeg', '-y', '-i', file_in, '-map', f'0:{i}', '-vn', file_out),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)

def convert_to_numpy(base_dir, target_dir):
    music_list = os.listdir(base_dir)
    music_list.sort()

    warnings.filterwarnings('ignore')
    for music in tqdm(music_list):
        music = os.path.join(base_dir, music)
        if not os.path.isdir(music):
            outfile_name = music.split("/")[-1]
            outfile_name = os.path.join(target_dir, outfile_name)
            arr, _ = librosa.load(music)
            np.save(outfile_name, arr)

def separate_segment(base_dir, target_dir, sep_length):
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