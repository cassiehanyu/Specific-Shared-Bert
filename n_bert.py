import torch
import torch.nn as nn
import copy

from pytorch_pretrained_bert import BertModel
from specific_shared import Encoder_Pooler


class nBert(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n = config.get('number')
        self.encoder_pooler_list = []

        bert = BertModel.from_pretrained('bert-base-chinese')
        self.embedding = bert.embeddings

        for i in range(self.n):
            self.encoder_pooler_list.append(Encoder_Pooler(copy.deepcopy(bert.encoder), copy.deepcopy(bert.pooler)))

        self.classifier = nn.Sequential(
            nn.Dropout(float(config['dropout'])),
            nn.Linear(n_feats * self.n, 2),
            # nn.LogSoftmax(1)
        )


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, label_tensor=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        feats = [bert_encoder_pooler(embedding_output, extended_attention_mask, output_all_encoded_layers=False)[1] for bert_encoder_pooler in encoder_pooler_list]

        feat_all = torch.cat(feats, dim=1)

        preds = self.classifier(feat_all)

        if label_tensor is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(preds, label_tensor)
            return loss
        else:
            return preds
