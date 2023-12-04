import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    assert len(dataset_items) > 0

    result_batch = dict()

    result_batch["audio_name"] = [item["audio_name"] for item in dataset_items]
    if 'text' in dataset_items[0]:
        result_batch["text"] = [item["text"] for item in dataset_items]

    spectrogram_length = [item["spectrogram"].size(2) for item in dataset_items]
    result_batch["spectrogram_length"] = torch.tensor(spectrogram_length)

    batch_size = len(dataset_items)
    spec_n_freq = dataset_items[0]["spectrogram"].size(1)
    max_spec_len = max(spectrogram_length)
    spectrogram = torch.full((batch_size, spec_n_freq, max_spec_len), 0.0)
    for i, item in enumerate(dataset_items):
        spectrogram[i, :, :spectrogram_length[i]] = item["spectrogram"]
    result_batch["spectrogram"] = spectrogram

    if 'path' in dataset_items[0]:
        result_batch["path"] = [item["path"] for item in dataset_items]

    if 'audio' in dataset_items[0]:
        audio_length = [item["audio"].size(1) for item in dataset_items]
        result_batch["audio_length"] = torch.tensor(audio_length)

        max_audio_len = max(audio_length)
        audio = torch.full((batch_size, max_audio_len), 0.0)
        for i, item in enumerate(dataset_items):
            audio[i, :audio_length[i]] = item["audio"][0]
        result_batch["audio"] = audio

    return result_batch
