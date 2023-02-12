import argparse
import os

import torch
import librosa
import soundfile
import numpy as np

from model.waveunet import WaveUNet

def main(args):
    path_model = os.path.expanduser(args.path_model)
    path_song = os.path.expanduser(args.path_song)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = WaveUNet().to(device)
    model.load_state_dict(torch.load(path_model))

    song_np, _ = librosa.load(path_song)
    song = torch.Tensor(song_np.reshape(1, 1, -1)).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(song)

    pred = pred.cpu()

    for i in range(4):
        soundfile.write(os.path.join(args.path_to_save, f'song_out_{i}.wav'), pred[0][i], 22050, format='wav')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict')

    parser.add_argument('--path_model', type=str, default='model.pt')
    parser.add_argument('--path_song', type=str, required=True)
    parser.add_argument('--path_to_save', type=str, default='./output')

    args = parser.parse_args()

    main(args)