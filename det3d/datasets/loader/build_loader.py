import platform
from functools import partial

from .collate_fns import collate
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, RandomSampler
import torch
import random
import numpy as np

from .sampler import (
    BatchSampler,
)

if platform.system() != "Windows":
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def worker_seed_set(worker_id):
    # See for details of numpy:
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    # See for details of random:
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    # NumPy
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))

    # random
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def build_dataloader(
        dataset, batch_size, workers_per_gpu, num_gpus=1, **kwargs
):
    shuffle = kwargs.get("shuffle", True)
    batch_size = num_gpus * batch_size
    num_workers = num_gpus * workers_per_gpu

    if shuffle:
        worker_func = worker_seed_set
    else:
        worker_func = None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        worker_init_fn=worker_func
    )

    return data_loader
