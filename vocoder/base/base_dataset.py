import logging
import random
from typing import List

import numpy as np
from scipy.io.wavfile import read
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from librosa.util import normalize

from vocoder.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


MAX_WAV_VALUE = 32768.0


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            wav2spec,
            config_parser: ConfigParser,
            slice_len=None,
            limit=None,
            max_audio_length=None,
    ):
        self.wav2spec = wav2spec
        self.config_parser = config_parser
        self.slice_len = slice_len

        # index = self._filter_records_from_dataset(index, max_audio_length, max_text_length, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        # index = self._sort_index(index)
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        sample = {"audio_name": data_dict["audio_name"]}
        if 'text' in data_dict:
            sample['text'] = data_dict["text"]

        if "path" in data_dict:
            audio_path = data_dict["path"]
            audio_wave = self.load_audio(audio_path)
            audio_wave = self._get_random_audio_slice(audio_wave)
            sample.update({
                "path": audio_path,
                "audio": audio_wave,
                "spectrogram": self.wav2spec(audio_wave).detach(),
                # "duration": audio_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            })
        else:  # inference
            sample.update({
                "spectrogram": torch.from_numpy(data_dict["spectrogram"]).unsqueeze(0),
            })

        return sample

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        '''
        Load audio from path, resample it if needed
        :return: 1st channel of audio (tensor of shape 1xL)
        '''

        # audio_tensor, sr = torchaudio.load(path)
        # audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        # target_sr = self.config_parser["preprocessing"]["sr"]
        # if sr != target_sr:
        #     audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        # return audio_tensor

        sampling_rate, audio = read(path)
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95
        return torch.FloatTensor(audio).unsqueeze(0)

    def _get_random_audio_slice(self, audio_wave):
        if self.slice_len is None or audio_wave.size(1) < self.slice_len:
            return audio_wave
        slice_start = random.randint(0, audio_wave.size(1) - self.slice_len)
        return audio_wave[:, slice_start:slice_start + self.slice_len]

    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, max_text_length, limit
    ) -> list:
        '''
        Filter records depending on max_audio_length, max_text_length and limit
        '''
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["audio_len"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        records_to_filter = exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            # random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index
