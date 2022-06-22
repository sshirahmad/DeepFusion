from __future__ import division
import numpy as np
from torch.utils.data import Sampler


class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last, multiscale_step=None, img_size=None):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1:
            raise ValueError("multiscale_step should be > 0, but got "
                             "multiscale_step={}".format(multiscale_step))

        self.multiscale_step = multiscale_step
        self.img_size = img_size
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32

    def __iter__(self):
        num_batch = 0
        batch = []
        size = self.img_size
        for idx in self.sampler:
            batch.append([idx, size])
            if len(batch) == self.batch_size:
                yield batch
                num_batch += 1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0:
                    size = np.random.choice(range(self.min_size, self.max_size + 1, 32))
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

