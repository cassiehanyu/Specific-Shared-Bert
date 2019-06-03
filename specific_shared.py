import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
# from core.models.BaseModel import BaseModel
# from core.utils.ModelFactory import ModelFactory
from pytorch_pretrained_bert import BertModel


"""
Reference: Adversarial Multi-task Learning for Text Classification
Link: https://arxiv.org/pdf/1704.05742.pdf
"""


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, input):
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradientReverse.apply(x)


class Encoder_Pooler(nn.Module):

    def __init__(self, encoder, pooler):
        super().__init__()
        self.encoder = encoder
        self.pooler = pooler


    def forward(self, embedding_output, extended_attention_mask, output_all_encoded_layers=True):
        encoded_layers = self.encoder(embedding_output,
                                    extended_attention_mask,
                                    output_all_encoded_layers=output_all_encoded_layers)

        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        return encoded_layers, pooled_output


class SpecificShared(nn.Module):

    def __init__(self, config):
        super().__init__()

        bert1 = BertModel.from_pretrained('bert-base-chinese').to("cuda")
        bert2 = BertModel.from_pretrained('bert-base-chinese').to("cuda")
        bert3 = BertModel.from_pretrained('bert-base-chinese').to("cuda")

        self.embeddings = bert1.embeddings

        self.generator = Encoder_Pooler(bert1.encoder, bert1.pooler)
        self.source_net = Encoder_Pooler(bert2.encoder, bert2.pooler)
        self.target_net = Encoder_Pooler(bert3.encoder, bert3.pooler)

        n_feats = 768
        num_classes = int(config['num_classes'])
        self.hidden_layer_units = int(config['hidden_layer_units'])

        self.discriminator = nn.Sequential(
            nn.Dropout(float(config['dropout'])),
            nn.Linear(n_feats, num_classes),
            nn.LogSoftmax(1)
        )

        self.last_layers_s = nn.Sequential(
            # nn.Linear(n_feats * 2, self.hidden_layer_units),
            # nn.Tanh(),
            nn.Dropout(float(config['dropout'])),
            nn.Linear(n_feats * 2, num_classes),
            # nn.LogSoftmax(1)
        )

        self.last_layers_t = nn.Sequential(
            # nn.Linear(n_feats * 2, self.hidden_layer_units),
            # nn.Tanh(),
            nn.Dropout(float(config['dropout'])),
            nn.Linear(n_feats * 2, num_classes),
            # nn.LogSoftmax(1)
        )

        # # print("generator:", list(self.generator.parameters())[37])
        # # print("target:", list(self.target_net.parameters())[37])
        # for i in range(194):
        #     print(all(list(self.generator.parameters())[1] == list(self.target_net.parameters())[1]))

        self.use_adv_loss = config.getboolean('use_adv_loss')
        self.use_orth_constraint = config.getboolean('use_orth_constraint')

        self.lamb = float(config['lambda'])
        self.gamma = float(config['gamma'])
        self.alpha = float(config['alpha'])

        self.register_buffer('source_label', torch.tensor(0))
        self.register_buffer('target_label', torch.tensor(1))


    def forward(self, source, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)


        _, feat_shared = self.generator(embedding_output, extended_attention_mask, output_all_encoded_layers=False)

        if source:
            _, feat_specific = self.source_net(embedding_output, extended_attention_mask, output_all_encoded_layers=False)
            # print('should be no source')
        else:
            _, feat_specific = self.target_net(embedding_output, extended_attention_mask, output_all_encoded_layers=False)

        feat_all = torch.cat([feat_specific, feat_shared], dim=1)

        if source:
            preds = self.last_layers_s(feat_all)
        else:
            preds = self.last_layers_t(feat_all)

        return preds, feat_shared, feat_specific
        

    def _adversarial_loss(self, feature, task_label, loss_fn):
        feature = grad_reverse(feature)

        preds = self.discriminator(feature)

        loss = loss_fn(preds, task_label.expand_as(preds[:, 0]))

        return loss


    def diff_loss(self, shared_feat, task_feat):
        task_feat_c = task_feat - torch.mean(task_feat, dim=0)
        task_feat_c = task_feat_c / torch.norm(task_feat_c, dim=1, keepdim=True)

        shared_feat_c = shared_feat - torch.mean(shared_feat, dim=0)
        shared_feat_c = shared_feat_c / torch.norm(shared_feat_c, dim=1, keepdim=True)

        dot_product = torch.mm(torch.transpose(task_feat_c, 0, 1), shared_feat_c)

        # average of squared norm
        constraint = torch.mean(torch.mul(dot_product, dot_product))

        return constraint


    # def _inter_domain_rel(self):
    #     w_sc = self.sc_weights[0].weight
    #     w_s = self.s_weights[0].weight

    #     w_t = self.t_weights[0].weight
    #     w_tc = self.tc_weights[0].weight

    #     new_shape = w_sc.shape[0] * w_sc.shape[1]

    #     w_sc = w_sc.reshape(new_shape, 1)
    #     w_s = w_s.reshape(new_shape, 1)
    #     w_tc = w_tc.reshape(new_shape, 1)
    #     w_t = w_t.reshape(new_shape, 1)

    #     w = torch.cat([w_sc, ws, w_tc, w_t], dim=1)
    #     loss = torch.mm(torch.mm(w, torch.inverse(self.omega)), torch.transpose(w))

    #     loss = torch.trace(loss)
    #     return loss


    def train_step(self, source, loss_fn, optimizer, tokens_tensor, segments_tensor, mask_tensor, label_tensor):
        preds, feat_shared, feat_specific = self.forward(source, tokens_tensor, segments_tensor, mask_tensor)

        optimizer.zero_grad()
        loss = loss_fn(preds, label_tensor)

        task_label = self.source_label if source else self.target_label

        if self.use_adv_loss:
            loss_adv = self._adversarial_loss(feat_shared, task_label, loss_fn)
            # loss_adv = self._adversarial_loss(feat_shared)
            loss += self.lamb * loss_adv

        if self.use_orth_constraint:
            constraint = self.diff_loss(feat_specific, feat_shared)

            loss += self.gamma * constraint


        loss.backward()
        optimizer.step()

        return loss, preds


    def test_step(self, tokens_tensor, segments_tensor, mask_tensor):
        preds, feat_shared, feat_specific = self.forward(False, tokens_tensor, segments_tensor, mask_tensor)
        return preds
