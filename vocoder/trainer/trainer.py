import itertools

import PIL
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from vocoder.base import BaseTrainer
from vocoder.logger.utils import plot_spectrogram_to_buf
from vocoder.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            g_criterion, d_criterion,
            g_optimizer, d_optimizer,
            g_lr_scheduler, d_lr_scheduler,
            metrics,
            config,
            device,
            dataloaders,
            wav2spec,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, g_criterion, d_criterion, g_optimizer, d_optimizer, metrics, config, device)
        self.skip_oom = skip_oom
        self.wav2spec = wav2spec
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.g_lr_scheduler = g_lr_scheduler
        self.d_lr_scheduler = d_lr_scheduler
        self.log_step = config["trainer"]["log_step"]

        self.train_metrics = MetricTracker(
            "disc_loss", "gen_loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "disc_loss", "gen_loss", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for key, value in batch.items():
            if isinstance(value, Tensor):
                batch[key] = batch[key].to(device)
        return batch

    def _clip_grad_norm(self, for_generator):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            if for_generator:
                params = self.model.generator.parameters()
            else:
                params = itertools.chain(self.model.msd.parameters(), self.model.mpd.parameters())

            clip_grad_norm_(params, self.config["trainer"]["grad_norm_clip"])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            # self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Discriminator Loss: {:.6f} Generator Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["d_loss"].item(), batch["g_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "generator learning rate", self.g_lr_scheduler.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "discriminator learning rate", self.d_lr_scheduler.get_last_lr()[0]
                )
                self._log(batch, self.train_metrics)

                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)

        # discriminator step
        if is_train:
            self.d_optimizer.zero_grad()
        outputs = self.model(**batch)
        batch.update(outputs)

        batch["d_loss"] = self.d_criterion(**batch)
        if is_train:
            batch["d_loss"].backward()
            self._clip_grad_norm(for_generator=False)
            self.d_optimizer.step()
            if self.d_lr_scheduler is not None:
                self.d_lr_scheduler.step()
        metrics.update("disc_loss", batch["d_loss"].item())

        # generator step
        if is_train:
            self.g_optimizer.zero_grad()
        outputs = self.model(**batch)
        batch.update(outputs)

        batch["g_loss"] = self.g_criterion(**batch)
        if is_train:
            batch["g_loss"].backward()
            self._clip_grad_norm(for_generator=True)
            self.g_optimizer.step()
            if self.g_lr_scheduler is not None:
                self.g_lr_scheduler.step()
        metrics.update("gen_loss", batch["g_loss"].item())

        # log metrics and return batch
        for met in self.metrics:
            metrics.update(met.name, met(**batch).item())
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log(batch, self.evaluation_metrics)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log(self, batch, metrics):
        self._log_scalars(metrics)
        self._log_spectrogram(batch["spectrogram"][0].detach().cpu(), 'spec')
        self._log_audio(batch["audio"][0], self.config["preprocessing"]["sr"], 'audio_target')
        self._log_audio(batch["gen_audio"][0], self.config["preprocessing"]["sr"], 'audio_pred')
        self._log_text(batch['text'][0])

    def _log_spectrogram(self, mel_spec, name):
        image = PIL.Image.open(plot_spectrogram_to_buf(mel_spec))
        self.writer.add_image(name, ToTensor()(image))

    def _log_audio(self, audio, sample_rate, name):
        self.writer.add_audio(name, audio, sample_rate)

    def _log_text(self, text):
        self.writer.add_text("text", text)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
