import argparse
from pathlib import Path

from librosa.util import normalize
import numpy as np
from scipy.io.wavfile import read
import torch

from vocoder.spectrogram import MelSpectrogram


ROOT_PATH = Path(__file__).absolute().resolve().parent.parent

MAX_WAV_VALUE = 32768.0


def main(audio_dir, mel_dir):
    wav2spec = MelSpectrogram()

    with torch.inference_mode():
        for audio_path in Path(audio_dir).iterdir():
            sampling_rate, audio = read(str(audio_path))
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            audio = torch.FloatTensor(audio).unsqueeze(0)
            mel = wav2spec(audio).squeeze(0).numpy()

            mel_filename = str(audio_path.stem) + '_mel.npy'
            mel_path = str(Path(mel_dir) / mel_filename)
            np.save(mel_path, mel)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "audio_dir",
        type=str,
        help="Directory to read audio from (default: None)",
    )
    args.add_argument(
        "mel_dir",
        type=str,
        help="Directory to save generated mel specs to (default: None)",
    )
    args = args.parse_args()

    main(args.audio_dir, args.mel_dir)
