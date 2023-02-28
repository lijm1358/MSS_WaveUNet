# MSS_WaveUNet

Music source separation using WaveUNet(https://arxiv.org/pdf/1806.03185.pdf)    

Separate 4 audio stems(vocal, drum, bass, others) from music.

## Setup

1. clone this repository
2. install requirements
```python
python -m pip install -r requirements.txt
```

3. install ffmpeg

for linux:
```bash
sudo apt install ffmpeg
```
  
for windows:  
go to ffmpeg installation webpage(https://ffmpeg.org/download.html#build-windows) and download ffmpeg executable file and add downloaded directory's path(`path\ffmpeg\bin`) to your environment variable `Path`.

## Train
You can train WaveUNet model with MUSDB18 dataset.
1. download musdb18 dataset(https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems)
2. unzip `musdb18.zip` or `musdb18hq.zip`
3. process the mp4 musdb dataset to numpy arrays. in `data` directory, you can use `process.py` to separate mp4 audio data into mp3 audio stems, and convert them into the same length array segments.
```bash
python process.py path/to/musdb18/dataset path/to/save
```
4. train the model using `train.py`. if you have processed the musdb18 dataset with `process.py`, the directory of the train dataset will be `path/to/save/train/data_split`, and the test dataset will be `path/to/save/test/data_split`.
```bash
python train.py path/train/data_split path/test/data_split
```
5. the best model(by validation loss, using early stopping) will be saved as `model.pt`. 
   
Or, you can use your own dataset to train the WaveUNet model, but you may need to use your own data processing method or script.

## Predict
You can separate the music into 4 stems using `predict.py`: vocal, drum, bass, and others.
Result of the prediction will be saved in `output/song_out_n.wav`. `n` will be the stem numbers, from 0 to 3, each means the drum, bass, others, and vocal.
```bash
python predict.py --path_model model.pt --path_song path/to/song.mp3
```

## Models
|Model   |Test sdr(accompanies) |Test sdr(vocal)   |File                    |
|--------|----------------------|------------------|------------------------|
|baseline|1.374189140248129     |0.9882581233978271|`checkpoint/baseline.pt`|

## Notes
This is just the trial of implementation of WaveUNet. The models are not may as good as the result of the original paper, or the other implementations.
