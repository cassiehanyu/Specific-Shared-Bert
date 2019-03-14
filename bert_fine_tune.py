import torch
import torch.nn as nn
import copy

from pytorch_pretrained_bert import BertModel
from loss_factory import LossFactory


class BertFineTune(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.loss = config['loss']
        n_feat = 768
        label_num = int(config["num_classes"])

        self.classifier = nn.Sequential(
            nn.Dropout(float(config['dropout'])),
            nn.Linear(n_feat, label_num)
        )


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, label_tensor=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        preds = self.classifier(pooled_output)

        batch_size = input_ids.size(0)

        if label_tensor is not None:
            loss_fn = LossFactory.get_lossfn(self.loss)()
            # loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(preds, label_tensor)
            return loss
        else:
            return preds