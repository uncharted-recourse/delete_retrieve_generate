""" global object so that everything knows whether we're on a gpu or not"""

import torch

torch.manual_seed(1)

CUDA = (torch.cuda.device_count() > 0)

