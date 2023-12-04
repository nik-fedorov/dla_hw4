import argparse
import collections
import itertools
import warnings

import numpy as np
import torch

import vocoder.loss as module_loss
import vocoder.metric as module_metric
import vocoder.model as module_arch
import vocoder.spectrogram as spectrogram_module
from vocoder.trainer import Trainer
from vocoder.utils import prepare_device
from vocoder.utils.object_loading import get_dataloaders
from vocoder.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    wav2spec_for_dataset = config.init_obj(config['preprocessing']['spectrogram'], spectrogram_module)
    dataloaders = get_dataloaders(config, wav2spec_for_dataset)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    wav2spec = config.init_obj(config['preprocessing']['spectrogram'], spectrogram_module).to(device)
    g_loss = config.init_obj(config["loss"]["g_loss"], module_loss, wav2spec=wav2spec)
    d_loss = config.init_obj(config["loss"]["d_loss"], module_loss)
    metrics = [
        config.init_obj(metric_dict, module_metric, wav2spec=wav2spec)
        for metric_dict in config["metrics"]
    ]

    g_optimizer = config.init_obj(config["optimizer"]["g_optimizer"], torch.optim, model.generator.parameters())
    d_optimizer = config.init_obj(config["optimizer"]["d_optimizer"], torch.optim,
                                  itertools.chain(model.msd.parameters(), model.mpd.parameters()))
    g_lr_scheduler = config.init_obj(config["lr_scheduler"]["g_lr_scheduler"], torch.optim.lr_scheduler, g_optimizer)
    d_lr_scheduler = config.init_obj(config["lr_scheduler"]["d_lr_scheduler"], torch.optim.lr_scheduler, d_optimizer)

    trainer = Trainer(
        model,
        g_loss, d_loss,
        g_optimizer, d_optimizer,
        g_lr_scheduler, d_lr_scheduler,
        metrics=metrics,
        wav2spec=wav2spec,
        config=config,
        device=device,
        dataloaders=dataloaders,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
