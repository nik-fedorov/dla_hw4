import torch
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mpd_gen_out, msd_gen_out, mpd_real_out, msd_real_out, **batch):
        loss = 0

        for dr, dg in zip(mpd_real_out, mpd_gen_out):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg ** 2)
            loss += (r_loss + g_loss)

        for dr, dg in zip(msd_real_out, msd_gen_out):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg ** 2)
            loss += (r_loss + g_loss)

        return loss
