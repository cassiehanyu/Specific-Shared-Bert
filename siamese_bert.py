import torch
import torch.nn as nn

import numpy as np

from pytorch_pretrained_bert import BertModel


class SiameseBert(nn.Module):

	def __init__(self, config):
		self.encoder = BertModel.from_pretrained('bert-base-chinese')


	def forward(self, tokens_tensor_left, segments_tensor_left, mask_tensor_left,
			tokens_tensor_right, segments_tensor_right, mask_tensor_right, 
			label_tensor=None, loss_fn):

		batch_size = tokens_tensor_left.size(0)
		_, hidden_left = encoder(tokens_tensor_left, segments_tensor_left, mask_tensor_left, output_all_encoded_layers=False)

		_, hidden_right = encoder(tokens_tensor_right, segments_tensor_right, mask_tensor_right, output_all_encoded_layers=False)

		result = torch.exp(-torch.norm(hidden_left - hidden_right), 1, dim=2)

		if label_tensor:
			loss = loss_fn(result.view(batch_size), label_tensor)
			return loss
		else:
			return result