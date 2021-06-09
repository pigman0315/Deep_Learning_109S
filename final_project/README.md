# LearningToProtect
Implementation of [Learning to Protect Communications with Adversarial Neural Cryptography](https://arxiv.org/abs/1610.06918) in PyTorch

## Requirements

`pip install -r requirements.txt` (PyTorch, Numpy, TensorboardX)

## Train

`python trainer.py -c config/default.yaml -n [name of run]`

- You may copy `cp config/default.yaml config/config.yaml` and change parameters (e.g. size of plain/key/cipher, depth of NN, …) to experiment with your own setting.

## Tensorboard

`tensorboard --logdir logs/`

