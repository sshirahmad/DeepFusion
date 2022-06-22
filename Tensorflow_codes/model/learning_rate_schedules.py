import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


class CosineAnnealer:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps

    def __call__(self, current_step):
        cos = np.cos(np.pi * (current_step / self.steps)) + 1

        return self.end + (self.start - self.end) / 2. * cos


class OneCycleScheduler(tf.optimizers.schedules.LearningRateSchedule):
    """ Callback
 that schedules the learning rate on a 1cycle policy as per Leslie Smith's paper(https://arxiv.org/pdf/1803.09820.pdf).
    If the model supports a momentum parameter, it will also be adapted by the schedule.
    The implementation adopts additional improvements as per the fastai library: https://docs.fast.ai/callbacks.one_cycle.html, where
    only two phases are used and the adaptation is done using cosine annealing.
    In phase 1 the LR increases from lrmax÷fac→r to lrmax and momentum decreases from mommax to mommin
.
    In the second phase the LR decreases from lrmax to lrmax÷fac→r⋅1e4 and momemtum from mommax to mommin
.
    By default the phases are not of equal length, with the phase 1 percentage controlled by the parameter phase1_pct
.
    """

    def __init__(self, lr_max, steps, wd_max=5e-4, mom_min=0.85, mom_max=0.95, pct=0.3, div_factor=25.):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor
        wd_min = wd_max / div_factor
        phase_1_steps = steps * pct
        phase_2_steps = steps - phase_1_steps
        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps

        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(wd_min, wd_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)],
                       [CosineAnnealer(lr_max, lr_min, phase_2_steps), CosineAnnealer(wd_max, wd_min, phase_1_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]

    def __call__(self, current_step):
        if current_step >= self.phase_1_steps:
            phase = 1
            current_step -= self.phase_1_steps
        else:
            phase = 0

        learning_rate = self.phases[phase][0](current_step)
        weight_decay = self.phases[phase][1](current_step)
        momentum = self.phases[phase][2](current_step)


        return learning_rate, weight_decay, momentum


class linear_warmup(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_step, warmup_init, max_lr, max_wd, training_step):
        self.warmup_step = warmup_step
        self.warmup_init = warmup_init
        self.max_lr = max_lr
        self.max_wd = max_wd
        self.training_step = training_step

    def __call__(self, current_step):
        if current_step < self.warmup_step:
            learning_rate = current_step / self.warmup_step * (self.max_lr - self.warmup_init) + self.warmup_init
            weight_decay = 0.0

        elif current_step < self.training_step // 3:
            learning_rate = self.max_lr
            weight_decay = self.max_wd
        elif current_step < self.training_step * 2 // 3:
            learning_rate = self.max_lr / 10.
            weight_decay = self.max_wd / 10.
        else:
            learning_rate = self.max_lr / 100.
            weight_decay = self.max_wd / 100.

        return learning_rate, weight_decay


class cosine_warmup(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_step, warmup_init, max_lr, max_wd, training_step):
        self.warmup_step = warmup_step
        self.warmup_init = warmup_init
        self.max_lr = max_lr
        self.max_wd = max_wd
        self.training_step = training_step

    def __call__(self, current_step):
        learning_rate = (0.5 * self.max_lr * (1 + np.cos(np.pi * float(current_step - self.warmup_step) / float(self.training_step - self.warmup_step))))
        weight_decay = (0.5 * self.max_wd * (1 + np.cos(np.pi * float(current_step - self.warmup_step) / float(self.training_step - self.warmup_step))))

        if self.warmup_step > 0:
            slope = (self.max_lr - self.warmup_init) / self.warmup_step
            pre_cosine_learning_rate = slope * float(current_step) + self.warmup_init
            if current_step < self.warmup_step:
                return pre_cosine_learning_rate, 0.0
            else:
                return learning_rate, weight_decay


def create_lr_scheduler(config, training_step):
    type = config.pop('type')

    if type == 'linear_warmup':
        return linear_warmup(warmup_step=config.warmup_step,
                             warmup_init=config.warmup_init,
                             max_lr=config.max_lr,
                             max_wd=config.max_wd,
                             training_step=training_step)
    elif type == 'cosine_warmup':
        return cosine_warmup(warmup_step=config.warmup_step,
                             warmup_init=config.warmup_init,
                             max_lr=config.max_lr,
                             max_wd=config.max_wd,
                             training_step=training_step)

    elif type == 'one_cycle':
        return OneCycleScheduler(lr_max=config.max_lr,
                                 steps=training_step,
                                 wd_max=config.wd,
                                 mom_min=config.mom_min,
                                 mom_max=config.mom_max,
                                 pct=config.pct,
                                 div_factor=config.div_factor)

    else:
        raise NotImplementedError

