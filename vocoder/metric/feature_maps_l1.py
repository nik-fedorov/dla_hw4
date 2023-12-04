from vocoder.base.base_metric import BaseMetric
from vocoder.loss import feature_loss


class FeatureMapsL1(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, **batch):
        loss_mpd = feature_loss(batch['mpd_gen_fmap'], batch['mpd_real_fmap'])
        loss_msd = feature_loss(batch['msd_gen_fmap'], batch['msd_real_fmap'])

        return loss_mpd + loss_msd
