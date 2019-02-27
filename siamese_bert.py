import torch
import torch.nn as nn

import numpy as np

from pytorch_pretrained_bert import BertModel


class SiameseBert(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-chinese')


    def _contrastive_loss(self, y, d, batch_size):
        tmp1 = y * torch.mul(d, d)
        tmp2 = (1-y)*torch.pow(torch.clamp((1-d), min=0.0), 2)
        return (tmp1 + tmp2)/batch_size/2


    def forward(self, tokens_tensor_left, segments_tensor_left, mask_tensor_left,
        tokens_tensor_right, segments_tensor_right, mask_tensor_right,
        label_tensor=None, loss_fn=None):

        batch_size = tokens_tensor_left.size(0)
        _, hidden_left = self.encoder(tokens_tensor_left, segments_tensor_left, mask_tensor_left, output_all_encoded_layers=False)

        _, hidden_right = self.encoder(tokens_tensor_right, segments_tensor_right, mask_tensor_right, output_all_encoded_layers=False)

        result = torch.norm((hidden_left - hidden_right), 2, dim=1)
        result /= (torch.norm(hidden_left, 2, dim=1) + torch.norm(hidden_right, 2, dim=1))

        if label_tensor is not None:
            loss_fn = self._contrastive_loss
            loss = loss_fn(result.view(batch_size), label_tensor, batch_size)
            return loss
        else:
            return result
