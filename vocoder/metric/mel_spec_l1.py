import torch.nn.functional as F

from vocoder.base.base_metric import BaseMetric


class MelSpecL1(BaseMetric):
    def __init__(self, wav2spec, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wav2spec = wav2spec

    def __call__(self, **batch):
        loss_mel = F.l1_loss(batch['spectrogram'],
                             self.wav2spec(batch['gen_audio'].squeeze()))
        return loss_mel
