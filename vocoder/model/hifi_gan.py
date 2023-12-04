from vocoder.base.base_model import BaseModel
from vocoder.model.generator import Generator
from vocoder.model.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator


class HiFiGAN(BaseModel):
    def __init__(self, generator_config):
        super().__init__()
        self.generator = Generator(generator_config)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, spectrogram, **batch):
        gen_audio = self.generator(spectrogram)
        if 'audio' not in batch:
            return {'gen_audio': gen_audio}

        audio = batch['audio']
        audio = audio.unsqueeze(1)

        real_len = audio.size(2)
        gen_audio = gen_audio[:, :, :real_len]

        mpd_gen_out, mpd_gen_fmap = self.mpd(gen_audio)
        msd_gen_out, msd_gen_fmap = self.msd(gen_audio)
        mpd_real_out, mpd_real_fmap = self.mpd(audio)
        msd_real_out, msd_real_fmap = self.msd(audio)

        return {
            'gen_audio': gen_audio,
            'mpd_gen_out': mpd_gen_out,
            'mpd_gen_fmap': mpd_gen_fmap,
            'msd_gen_out': msd_gen_out,
            'msd_gen_fmap': msd_gen_fmap,
            'mpd_real_out': mpd_real_out,
            'mpd_real_fmap': mpd_real_fmap,
            'msd_real_out': msd_real_out,
            'msd_real_fmap': msd_real_fmap,
        }
