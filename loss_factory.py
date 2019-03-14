import torch.nn as nn

class LossFactory(object):
    loss_map = {
        'nll': nn.NLLLoss,
        'mse': nn.MSELoss,
        'cross_entropy': nn.CrossEntropyLoss
    }

    @staticmethod
    def get_lossfn(loss_name):
        loss_map = LossFactory.loss_map
        return loss_map[loss_name]
