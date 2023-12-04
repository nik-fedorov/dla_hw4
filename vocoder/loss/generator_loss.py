import torch
import torch.nn as nn
import torch.nn.functional as F


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss


def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((1-dg)**2)
    return loss


class GeneratorLoss(nn.Module):
    def __init__(self, lambda_fm, lambda_mel, wav2spec):
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        self.wav2spec = wav2spec

    def forward(self, **batch):
        loss_mel = F.l1_loss(batch['spectrogram'],
                             self.wav2spec(batch['gen_audio'].squeeze()))

        loss_fm_f = feature_loss(batch['mpd_gen_fmap'], batch['mpd_real_fmap'])
        loss_fm_s = feature_loss(batch['msd_gen_fmap'], batch['msd_real_fmap'])

        loss_gen_f = generator_loss(batch['mpd_gen_out'])
        loss_gen_s = generator_loss(batch['msd_gen_out'])

        loss = loss_gen_s + loss_gen_f
        loss += (loss_fm_s + loss_fm_f) * self.lambda_fm
        loss += loss_mel * self.lambda_mel

        return loss
