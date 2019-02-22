from tqdm import tqdm
import random 
import os
import numpy as np

import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, BertForNextSentencePrediction
from pytorch_pretrained_bert.optimization import BertAdam
from specific_shared import SpecificShared
from siamese_bert import SiameseBert


def load_pretrained_model_tokenizer(model_type="BertForSequenceClassification", device="cuda", config=None):
    # Load pre-trained model (weights)
    if model_type == "BertForSequenceClassification":
        model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
        # Load pre-trained model tokenizer (vocabulary)
    elif model_type == "BertForNextSentencePrediction":
        model = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')
    elif model_type == "specific_shared":
        model = SpecificShared(config)
    elif model_type == "siamese_bert":
        model = SiameseBert(config)
    else:
        print("[Error]: unsupported model type")
        return None, None
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model.to(device)
    print("Initialized model and tokenizer")
    return model, tokenizer


def load_data(data_path, dataset, data_name, batch_size, tokenizer, device="cuda"):
    f = open(os.path.join(data_path, "{}/{}.csv".format(dataset, data_name)))
    test_batch, testid_batch, mask_batch, label_batch = [], [], [], []
    data_set = []
    for l in f:
        label, a, b = l.replace("\n", "").split("\t")
        a_index = tokenize_index(a, tokenizer)
        b_index = tokenize_index(b, tokenizer)
        combine_index = a_index + b_index
        segments_ids = [0] * len(a_index) + [1] * len(b_index)
        test_batch.append(torch.tensor(combine_index))
        testid_batch.append(torch.tensor(segments_ids))
        mask_batch.append(torch.ones(len(combine_index)))
        label_batch.append(int(label))
        if len(test_batch) >= batch_size:
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.nn.utils.rnn.pad_sequence(test_batch, batch_first=True, padding_value=0).to(device)
            segments_tensor = torch.nn.utils.rnn.pad_sequence(testid_batch, batch_first=True, padding_value=0).to(device)
            mask_tensor = torch.nn.utils.rnn.pad_sequence(mask_batch, batch_first=True, padding_value=0).to(device)
            label_tensor = torch.tensor(label_batch, device=device)
            data_set.append((tokens_tensor, segments_tensor, mask_tensor, label_tensor))
            test_batch, testid_batch, mask_batch, label_batch = [], [], [], []

    if len(test_batch) != 0:
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.nn.utils.rnn.pad_sequence(test_batch, batch_first=True, padding_value=0).to(device)
        segments_tensor = torch.nn.utils.rnn.pad_sequence(testid_batch, batch_first=True, padding_value=0).to(device)
        mask_tensor = torch.nn.utils.rnn.pad_sequence(mask_batch, batch_first=True, padding_value=0).to(device)
        label_tensor = torch.tensor(label_batch, device=device)
        data_set.append((tokens_tensor, segments_tensor, mask_tensor, label_tensor))
        test_batch, testid_batch, mask_batch, label_batch = [], [], [], []
    
    print('finish loading dataset', data_name)
    return data_set

def init_optimizer(model, learning_rate, warmup_proportion, num_train_epochs, data_size):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    num_train_steps = data_size * num_train_epochs
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
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
    if len(result.shape) > 1:
        pred = np.argmax(predictions, axis=1)
    else:
        pred = np.rint(predictions)
    return list(pred)


def get_predicted_score(predictions):
    ret = predictions
    if len(predictions.shape) > 1:
        if predictions.shape[1] == 2:
            ret = predictions[:, 1]
        elif predictions.shape[1] == 1:
            ret = predictions[:, 0]
        elif predictions.shape[1] == 3:
            ret = predictions[:, 2]
    return list(ret)
