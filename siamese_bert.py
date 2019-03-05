import torch
import torch.nn as nn

import numpy as np

from pytorch_pretrained_bert import BertModel


class SiameseBert(nn.Module):

    def __init__(self, config):
        super().__init__()
        n_feats = 768

        self.encoder = BertModel.from_pretrained('bert-base-chinese')
        # self.decoder = BertModel.from_pretrained('bert-base-chinese')
        self.dnn = nn.Sequential(
            nn.Linear(n_feats, n_feats),
            nn.Linear(n_feats, n_feats),
            nn.Linear(n_feats, n_feats)
        )

    def _contrastive_loss(self, d, y, batch_size):
        tmp1 = y * torch.pow(d, 2)
        tmp2 = (1-y)*torch.pow(torch.clamp((1-d), min=0.0), 2)
        return torch.sum(tmp1 + tmp2)/batch_size/2.0


    def forward(self, tokens_tensor_left, segments_tensor_left, mask_tensor_left,
        tokens_tensor_right, segments_tensor_right, mask_tensor_right,
        label_tensor=None, loss_fn=None):

        batch_size = tokens_tensor_left.size(0)
        _, hidden_left = self.encoder(tokens_tensor_left, segments_tensor_left, mask_tensor_left, output_all_encoded_layers=False)

        _, hidden_right = self.encoder(tokens_tensor_right, segments_tensor_right, mask_tensor_right, output_all_encoded_layers=False)

        hidden_right_prime = self.dnn(hidden_right)

        # result = torch.norm((hidden_left - hidden_right), 2, dim=1)
        # result /= (torch.norm(hidden_left, 2, dim=1) + torch.norm(hidden_right, 2, dim=1))

        dot_prod = torch.sum(hidden_left * hidden_right_prime, dim=-1).reshape(batch_size, 1)

        zero_prob = 1 - dot_prod

        logits = torch.cat([zero_prob, dot_prod], dim=1)

        if label_tensor is not None:
            # loss_fn = self._contrastive_loss
            loss = self.loss(logits, label_tensor)
            return loss
        else:
            return nn.Softmax()(logits)
            # return torch.exp(-result)
