import logging
from pathlib import Path

import numpy as np

from vocoder.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomDirSpectrogramDataset(BaseDataset):
    def __init__(self, spec_dir, transcription_dir=None, *args, **kwargs):
        data = []
        for path in Path(spec_dir).iterdir():
            entry = {}

            entry["spectrogram"] = np.load(str(path))
            entry["audio_name"] = str(path.stem)

            if transcription_dir and Path(transcription_dir).exists():
                transc_path = Path(transcription_dir) / (path.stem + '.txt')
                if transc_path.exists():
                    with transc_path.open() as f:
                        entry["text"] = f.read().strip()

            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
