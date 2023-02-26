import os

import numpy as np
from torch.utils.data import Dataset


class MUSDBDataset(Dataset):
    """MUSDB18 Dataset.

    Args:
        data_dir: directory where the same length segments of MUSDB18 are stored

    Attributes:
        music_list: list of names of segment
        data_dir: directory where the same length segments of MUSDB18 are stored
    """

    def __init__(self, data_dir: str):
        self.music_list = []
        self.data_dir = os.path.join(data_dir)
        self.music_list = self.get_filenames(self.data_dir)

    def __len__(self):
        return len(self.music_list)

    def __getitem__(self, idx):
        music = self.music_list[idx]
        music = np.load(music)
        return np.expand_dims(music[0], 0), music[1:]

    def get_filenames(self, path):
        files_list = list()
        filenames = os.listdir(path)
        filenames.sort()
        for filename in filenames:
            files_list.append(os.path.join(path, filename))
        return files_list
