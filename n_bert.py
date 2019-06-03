import torch
import torch.nn as nn
import copy

from pytorch_pretrained_bert import BertModel
from specific_shared import Encoder_Pooler
from loss_factory import LossFactory


class nBert(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n = int(config['number'])
        self.loss = config['loss']

        bert_model = config['bert_model']
        bert = BertModel.from_pretrained(bert_model)
        # bert2 = BertModel.from_pretrained('bert-base-chinese')
        bert3 = BertModel.from_pretrained(bert_model)
        self.embeddings = bert.embeddings

        n_feats = 768
        label_num = int(config["num_classes"])

        # ep_list = []
        # for i in range(self.n):
        #     tmp_bert = BertModel.from_pretrained("bert-base-chinese")
        #     ep_list.append(Encoder_Pooler(tmp_bert.encoder, tmp_bert.pooler).to("cuda"))

        # self.encoder_pooler_list = nn.ModuleList(ep_list)

        self.ep1 = Encoder_Pooler(bert.encoder, bert.pooler)
        # self.ep2 = Encoder_Pooler(bert2.encoder, bert2.pooler)
        self.ep3 = Encoder_Pooler(bert3.encoder, bert3.pooler)

        # self.encoder_pooler_list = [self.ep1, self.ep2, self.ep3]

        self.classifier = nn.Sequential(
            nn.Dropout(float(config['dropout'])),
            nn.Linear(n_feats * self.n, label_num),
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

        _, feat1 = self.ep1(embedding_output, extended_attention_mask, output_all_encoded_layers=False)
        _, feat2 = self.ep3(embedding_output, extended_attention_mask, output_all_encoded_layers=False)

        # feats = [feat1, feat2]

        # feats = []
        # for bert_encoder_pooler in self.encoder_pooler_list:
        #     _, feat = bert_encoder_pooler(embedding_output, extended_attention_mask, output_all_encoded_layers=False)
        #     feats.append(feat)

        feat_all = torch.cat([feat2, feat1], dim=1)

        preds = self.classifier(feat_all)

        if label_tensor is not None:
            loss_fn = LossFactory.get_lossfn(self.loss)()
            loss = loss_fn(preds, label_tensor)
            return loss
        else:
            return preds
