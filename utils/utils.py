import soundfile

def write_soundfile(filename, arr):
    soundfile.write(f".\\{filename}.wav", arr, 22050, format='wav')