from dataset.Inria.inria import Inria as inria_pd


def create_data_loader(config):
    type = config.type

    if type == 'inria':
        return inria_pd

    else:
        raise NotImplementedError
