import torch
import torch.nn as nn
import copy

from pytorch_pretrained_bert import BertModel


class BertSts(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = float(config['dropout'])

        n_feat = 768

        self.classifier = nn.Sequential(
            nn.Dropout(float(config['dropout'])),
            nn.Linear(n_feat, 1)
        )


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, label_tensor=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        preds = self.classifier(pooled_output)

        batch_size = input_ids.size(0)

        if label_tensor is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds.view(batch_size), label_tensor)
            return loss
        else:
            return preds
