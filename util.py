from tqdm import tqdm
import random
import os
import numpy as np

import torch
import torch.nn as nn
from scipy.stats import spearmanr
from scipy.special import softmax

from pydoc import locate

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, \
    BertForNextSentencePrediction
from pytorch_pretrained_bert.optimization import BertAdam
from specific_shared import SpecificShared
from siamese_bert import SiameseBert
from n_bert import nBert
from bert_sts import BertSts
from bert_fine_tune import BertFineTune
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)


def load_pretrained_model_tokenizer(model_type="BertForSequenceClassification", device="cuda", config=None):
    bert_model = config['bert_model']
    # Load pre-trained model (weights)
    if model_type == "BertForSequenceClassification":
        model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=2)
        # Load pre-trained model tokenizer (vocabulary)
    elif model_type == "BertForNextSentencePrediction":
        model = BertForNextSentencePrediction.from_pretrained(bert_model)
    elif model_type == "specific_shared":
        model = SpecificShared(config)
    elif model_type == "siamese_bert":
        model = SiameseBert(config)
    elif model_type == "n_bert":
        model = nBert(config)
    elif model_type == "bert_sts":
        model = BertSts(config)
    elif model_type == "bert_fine_tune":
        model = BertFineTune(config)
    else:
        print("[Error]: unsupported model type")
        return None, None

    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model.to(device)
    print("Initialized model and tokenizer")
    return model, tokenizer


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def tokenize_one(text, tokenizer):
    max_length = 50
    tokens = tokenizer.tokenize(text)

    if len(tokens) > max_length - 2:
        tokens = tokens[:(max_length - 2)]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segments_ids = [0] * len(tokens)

    combine_index = tokenizer.convert_tokens_to_ids(tokens)

    return combine_index, segments_ids


def tokenize_two(text_a, text_b, tokenizer):
    max_length = 50
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b)

    truncate_seq_pair(tokens_a, tokens_b, max_length - 3)

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    segments_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

    combine_index = tokenizer.convert_tokens_to_ids(tokens)

    return combine_index, segments_ids


def load_data(data_path, dataset, data_name, batch_size, tokenizer, device="cuda", label_type='int'):
    f = open(os.path.join(data_path, "{}/{}.csv".format(dataset, data_name)))
    test_batch, testid_batch, mask_batch, label_batch = [], [], [], []
    for l in f:
        data = l.replace("\n", "").split("\t")

        if len(data) == 3:
            label, a, b = data
            combine_index, segments_ids = tokenize_two(a, b, tokenizer)
        elif len(data) == 2:
            label, a = data
            combine_index, segments_ids = tokenize_one(a, tokenizer)

        mask_ids = [1] * len(combine_index)
        pad = [0] * (50 - len(combine_index))
        combine_index += pad
        segments_ids += pad
        mask_ids += pad

        test_batch.append(combine_index)
        testid_batch.append(segments_ids)
        mask_batch.append(mask_ids)
        label_batch.append(locate(label_type)(label))

    tokens_tensor = torch.tensor(test_batch, device=device, dtype=torch.long)
    segments_tensor = torch.tensor(testid_batch, device=device, dtype=torch.long)
    mask_tensor = torch.tensor(mask_batch, device=device, dtype=torch.long)
    label_tensor = torch.tensor(label_batch, device=device, dtype=torch.long)

    data_set = TensorDataset(tokens_tensor, segments_tensor, mask_tensor, label_tensor)

    print('finish loading dataset', data_name)
    return data_set


def load_data2(data_path, dataset, data_name, batch_size, tokenizer, device="cuda"):
    f = open(os.path.join(data_path, "{}/{}.csv".format(dataset, data_name)))
    test_batch_left, testid_batch_left, mask_batch_left = [], [], []
    test_batch_right, testid_batch_right, mask_batch_right = [], [], []
    label_batch = []
    data_set = []

    for l in f:
        label, a, b = l.replace("\n", "").split("\t")
        a_index = tokenize_index(a, tokenizer)
        b_index = tokenize_index(b, tokenizer)

        a_segments_ids = [0] * len(a_index)
        b_segments_ids = [0] * len(b_index)

        test_batch_left.append(torch.tensor(a_index))
        test_batch_right.append(torch.tensor(b_index))

        testid_batch_left.append(torch.tensor(a_segments_ids))
        testid_batch_right.append(torch.tensor(b_segments_ids))

        mask_batch_left.append(torch.ones(len(a_index)))
        mask_batch_right.append(torch.ones(len(b_index)))

        label_batch.append(float(label))

        if len(test_batch_left) >= batch_size:
            # Convert inputs to PyTorch tensors
            tokens_tensor_left = torch.nn.utils.rnn.pad_sequence(test_batch_left, batch_first=True, padding_value=0).to(
                device)
            tokens_tensor_right = torch.nn.utils.rnn.pad_sequence(test_batch_right, batch_first=True,
                                                                  padding_value=0).to(device)

            segments_tensor_left = torch.nn.utils.rnn.pad_sequence(testid_batch_left, batch_first=True,
                                                                   padding_value=0).to(device)
            segments_tensor_right = torch.nn.utils.rnn.pad_sequence(testid_batch_right, batch_first=True,
                                                                    padding_value=0).to(device)

            mask_tensor_left = torch.nn.utils.rnn.pad_sequence(mask_batch_left, batch_first=True, padding_value=0).to(
                device)
            mask_tensor_right = torch.nn.utils.rnn.pad_sequence(mask_batch_right, batch_first=True, padding_value=0).to(
                device)

            label_tensor = torch.tensor(label_batch, device=device)

            data_set.append((tokens_tensor_left, segments_tensor_left, mask_tensor_left,
                             tokens_tensor_right, segments_tensor_right, mask_tensor_right, label_tensor))

            test_batch_left, testid_batch_left, mask_batch_left = [], [], []
            test_batch_right, testid_batch_right, mask_batch_right = [], [], []

            label_batch = []

    if len(test_batch_left) != 0:
        # Convert inputs to PyTorch tensors
        tokens_tensor_left = torch.nn.utils.rnn.pad_sequence(test_batch_left, batch_first=True, padding_value=0).to(
            device)
        tokens_tensor_right = torch.nn.utils.rnn.pad_sequence(test_batch_right, batch_first=True, padding_value=0).to(
            device)

        segments_tensor_left = torch.nn.utils.rnn.pad_sequence(testid_batch_left, batch_first=True, padding_value=0).to(
            device)
        segments_tensor_right = torch.nn.utils.rnn.pad_sequence(testid_batch_right, batch_first=True,
                                                                padding_value=0).to(device)

        mask_tensor_left = torch.nn.utils.rnn.pad_sequence(mask_batch_left, batch_first=True, padding_value=0).to(
            device)
        mask_tensor_right = torch.nn.utils.rnn.pad_sequence(mask_batch_right, batch_first=True, padding_value=0).to(
            device)

        label_tensor = torch.tensor(label_batch, device=device)

        data_set.append((tokens_tensor_left, segments_tensor_left, mask_tensor_left,
                         tokens_tensor_right, segments_tensor_right, mask_tensor_right, label_tensor))

        test_batch_left, testid_batch_left, mask_batch_left = [], [], []
        test_batch_right, testid_batch_right, mask_batch_right = [], [], []

        label_batch = []

    print('finish loading dataset', data_name)
    return data_set


def init_optimizer(model, learning_rate, warmup_proportion, num_train_epochs, data_size, freeze_bert_layer=False):
    param_optimizer = list(model.named_parameters())
    freeze_list = ['embeddings']
    for i in range(8):
        freeze_list.append(f'layer.{i}.')
    if freeze_bert_layer:
        for name, param in param_optimizer:
            if any(x in name for x in freeze_list):
                param.requires_grad = False
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    num_train_steps = data_size * num_train_epochs
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=num_train_steps)

    return optimizer


def init_loss():
    # return nn.NLLLoss()
    return nn.CrossEntropyLoss()


def tokenize_index(text, tokenizer):
    tokenized_text = tokenizer.tokenize(text)
    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    return indexed_tokens


def get_acc(prediction_index_list, labels):
    acc = sum(np.array(prediction_index_list) == np.array(labels))
    return acc / len(labels)


def get_pre_rec_f1(prediction_index_list, labels):
    tp, tn, fp, fn = 0, 0, 0, 0
    for p, l in zip(prediction_index_list, labels):
        if p == l:
            if p == 1:
                tp += 1
            else:
                tn += 1
        else:
            if p == 1:
                fp += 1
            else:
                fn += 1
    eps = 1e-8
    precision = tp * 1.0 / (tp + fp + eps)
    recall = tp * 1.0 / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def get_p1(prediction_score_list, labels, data_path, dataset, data_name):
    f = open(os.path.join(data_path, "{}/{}.csv".format(dataset, data_name)))
    a2score_label = {}
    for line, p, l in zip(f, prediction_score_list, labels):
        label, a, b = line.replace("\n", "").split("\t")
        if a not in a2score_label:
            a2score_label[a] = []
        a2score_label[a].append((p, l))

    acc = 0
    no_true = 0
    for a in a2score_label:
        a2score_label[a] = sorted(a2score_label[a], key=lambda x: x[0], reverse=True)
        if a2score_label[a][0][1] > 0:
            acc += 1
        if sum([tmp[1] for tmp in a2score_label[a]]) == 0:
            no_true += 1

    p1 = acc / (len(a2score_label) - no_true)

    return p1


def get_predicted_index(predictions):
    if len(predictions.shape) > 1:
        pred = np.argmax(predictions, axis=1)
    else:
        pred = np.rint(predictions)
    return list(pred)


def get_predicted_score(predictions):
    predictions = softmax(predictions, axis=1)
    ret = predictions
    if len(predictions.shape) > 1:
        if predictions.shape[1] == 2:
            ret = predictions[:, 1]
        elif predictions.shape[1] == 1:
            ret = predictions[:, 0]
        elif predictions.shape[1] == 3:
            ret = predictions[:, 2]
    return list(ret)


def get_pearsonr(pred, label):
    return np.corrcoef(pred, label)[0, 1]


def get_spearmanr(pred, label):
    return spearmanr(pred, label)[0]
