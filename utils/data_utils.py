import os
import torch
import librosa
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset

class WAVDataset(Dataset):
    def __init__(self, root_dir, sample_rate, slice_len):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.slice_len = slice_len
        self.samples = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, id):
        x = load_wav(self.samples[id], sample_rate=self.sample_rate)
        if len(x) >= self.slice_len:
            return x[:, self.slice_len]
        else:
            return F.pad(x, pad=(0, self.slice_len - x.shape[1]))


def load_wav(wav_file_path, sample_rate):
    """
    Loads .wav into torch tensor of shape ([1, audio_len]).

    :param wav_file_path: path to .wav file
    :param sample_rate: sample rat of audio
    :return: waveform torch array
    """
    audio_data, _ = librosa.load(wav_file_path, sr=sample_rate)
    audio_data = np.reshape(audio_data, (1, -1))
    audio_data = torch.from_numpy(audio_data)
    return audio_data


def fix_length(fix_len):
    """
    Returns a function that fixes the length of a 1D array to fix_len.
    :param fix_len: length for array to be fixed to
    :return: function that fixes length
    """
    return lambda x: x[:, :fix_len] if len(x) >= fix_len \
           else F.pad(x, pad=(0, fix_len - x.shape[1]))

if __name__ == "__main__":
    data_folder = Path("test/data/")
    dataset = WAVDataset(data_folder, slice_len=32768)