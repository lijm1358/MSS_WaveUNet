import argparse
import os

import librosa
import numpy as np
import soundfile
import torch

from model.waveunet import WaveUNet


def main(args):
    path_model = os.path.expanduser(args.path_model)
    path_song = os.path.expanduser(args.path_song)
    path_to_save = os.path.normpath(args.path_to_save)

    os.makedirs(path_to_save, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = WaveUNet(n_source=2).to(device)
    model.load_state_dict(torch.load(path_model))

    song_np, _ = librosa.load(path_song)
    song = torch.Tensor(song_np.reshape(1, 1, -1)).to(device)

    model.eval()
    pred = None
    print(song.shape[2])
    with torch.no_grad():
        sep_length = 16384
        start = 0
        while start+sep_length < song.shape[2]:
            if pred == None:
                pred = model(song[:, :, start : start + sep_length])
            else:
                pred = torch.cat(
                    (pred, model(song[:, :, start : start + sep_length])), 2
                )
            start += sep_length

    pred = pred.cpu()

    for i in range(2):
        soundfile.write(
            os.path.join(path_to_save, f"song_out_{i}.wav"),
            pred[0][i],
            22050,
            format="wav",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict")

    parser.add_argument("--path_model", type=str, default="model.pt")
    parser.add_argument("--path_song", type=str, required=True)
    parser.add_argument("--path_to_save", type=str, default="./output")

    args = parser.parse_args()

    main(args)
