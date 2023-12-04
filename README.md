# Vocoder

## Report

About all details about this homework can be found in 
[wandb report](https://wandb.ai/nik-fedorov/dla_hw4/reports/Vocoder-HW4--Vmlldzo2MTYxNjU0). 

## Description

This is a repository containing a convenient pipeline for training vocoder (mel spectrograms to speech) models. 

Advantages of this repo:
- possibility of changing experimental configuration by only tuning one json file
- good and clean code structure (see `vocoder` folder with all elements of pipeline)
- prepared scripts for training and evaluation of models
- logging spectrograms, texts and audios (target and predicted) during training
- prepared downloadable checkpoint with high perfomance

## Installation guide

To set up the environment for this repository run the following command in your terminal (with your virtual environment activated):

```shell
pip install -r ./requirements.txt
```

## Evaluate model

To download my best checkpoint run the following:
```shell
python default_test_model/download_best_ckpt.py
```
if you are interested how I got this checkpoint, you can read about that in 
[wandb report](https://wandb.ai/nik-fedorov/dla_hw4/reports/Vocoder-HW4--Vmlldzo2MTYxNjU0).

You can evaluate model using `test.py` script. Here is an example of command to run my best checkpoint with default test config:

```shell
python test.py \
  -c default_test_model/config.json \
  -r default_test_model/checkpoint.pth \
  -t test_data/mels \
  -o path/to/your/output/directory
```

## Training
Use `train.py` for training. Example of command to launch training from scratch:
```shell
python train.py -c vocoder/configs/train_config.json
```

To fine-tune your checkpoint you can use option `-r` to pass path to the checkpoint file:
```shell
python train.py \
  -c vocoder/configs/train_config.json \
  -r saved/models/<exp name>/<run name>/checkpoint.pth
```

## Credits

This repository is based on a [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repository.
