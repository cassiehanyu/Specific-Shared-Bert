from tqdm import tqdm
import random 
import os 
import numpy as np
import argparse
import configparser
from specific_shared import SpecificShared

import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef 

from util import *

RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def print_config(config):
    for section in config.sections():
        for param in config[section]:
            print(param + "\t" + config[section][param])


def train(args, config):
    if args.load_trained:
        epoch, arch, model, tokenizer, scores = load_checkpoint(config['model_path']) 
    else:
        model, tokenizer = load_pretrained_model_tokenizer(config['model_type'], device=args.device, config=config)

    if config['model_type'] == "siamese_bert":
        train_dataset = load_data2(config['data_path'], config['dataset'], config['train'],
            int(config['batch_size']), tokenizer, args.device, label_type=config['label_type'])

        validate_dataset = load_data2(config['data_path'], config['dataset'], config['validate'],
            int(config['batch_size']), tokenizer, args.device, label_type=config['label_type'])

        test_dataset = load_data2(config['data_path'], config['dataset'], config['test'],
            int(config['batch_size']), tokenizer, args.device, label_type=config['label_type'])
    else:
        train_dataset = load_data(config['data_path'], config['dataset'], config['train'],
            int(config['batch_size']), tokenizer, args.device, label_type=config['label_type'])

        validate_dataset = load_data(config['data_path'], config['dataset'], config['validate'],
            int(config['batch_size']), tokenizer, args.device, label_type=config['label_type'])

        test_dataset = load_data(config['data_path'], config['dataset'], config['test'],
            int(config['batch_size']), tokenizer, args.device, label_type=config['label_type'])

    optimizer = init_optimizer(model, float(config['learning_rate']), float(config['warmup_proportion']), int(config['train_epoch']), len(train_dataset))

    model.train()
    global_step = 0
    best_score = 0
    model_path = ""
    for epoch in range(1, int(config['train_epoch'])+1):
        tr_loss = 0
        random.shuffle(train_dataset)
        for step, batch in enumerate(tqdm(train_dataset)):
            # tokens_tensor, segments_tensor, mask_tensor, label_tensor = batch
            if config['model_type'] == "siamese_bert":
                loss_fn = nn.MSELoss()
                loss = model(*batch, loss_fn)
            else:
                loss = model(*batch)

            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            model.zero_grad()
            global_step += 1
            
            if epoch == 1 and args.eval_steps > 0 and step % args.eval_steps == 0:
                best_score = eval_select(args, config, model, tokenizer, validate_dataset, test_dataset, config['model_path'], best_score, epoch, config['model_type'])

        print("[train] loss: {}".format(tr_loss))
        best_score = eval_select(args, config, model, tokenizer, validate_dataset, test_dataset, config['model_path'], best_score, epoch, config['model_type'])

    scores, _ = test(args, config,  split="test")
    print_scores(scores)


def transfer(args, config):
    if args.load_trained:
        epoch, arch, model, tokenizer, scores = load_checkpoint(args.pytorch_dump_path)
    else:
        model, tokenizer = load_pretrained_model_tokenizer(config['model_type'], device=args.device, config=config)

    s_train_dataset = load_data(config['data_path'], config['dataset'], config['s_train'], int(config['s_batch_size']), tokenizer, args.device)
    t_train_dataset = load_data(config['data_path'], config['dataset'], config['t_train'], int(config['t_batch_size']), tokenizer, args.device)

    validate_dataset = load_data(config['data_path'], config['dataset'], config['validate'], int(config['t_batch_size']), tokenizer, args.device)
    test_dataset = load_data(config['data_path'], config['dataset'], config['test'], int(config['t_batch_size']), tokenizer, args.device)

    print(len(t_train_dataset))

    optimizer = init_optimizer(model, float(config['learning_rate']), float(config['warmup_proportion']), int(config['train_epoch']), len(t_train_dataset))
    loss_fn = init_loss()

    best_score = 0

    global_step = 0
    for epoch in range(1, int(config['train_epoch'])+1):
        model.train()

        batches = []
        class_labels = []

        tr_loss = 0
        # random.shuffle(s_train_dataset)
        random.shuffle(t_train_dataset)

        # while s_train_dataset and t_train_dataset:
        #     next_from = np.random.choice(2, 1, p=[float(config['alpha']), 1-float(config['alpha'])])
        #     next_train_batch = s_train_dataset.pop(0) if next_from else t_train_dataset.pop(0)
        #     batches.append((next_train_batch, next_from))

        for step, batch in enumerate(t_train_dataset):
            next_from = 0
            tokens_tensor, segments_tensor, mask_tensor, label_tensor = batch
            # s_train_dataset.append(batch) if next_from else t_train_dataset.append(batch)

            loss, preds = model.train_step(next_from, loss_fn, optimizer, tokens_tensor, segments_tensor, mask_tensor, label_tensor)

            tr_loss += loss.item()
            global_step += 1

            if epoch == 1 and args.eval_steps > 0 and step % args.eval_steps == 0:
                best_score = eval_select(args, config, model, tokenizer, validate_dataset, test_dataset, config['model_path'], best_score, epoch, config['model_type'])
                print("[train] batch {}, loss: {}, best score: {}".format(step, loss, best_score))

        print("[train] loss: {}".format(tr_loss))
        best_score = eval_select(args, config, model, tokenizer, validate_dataset, test_dataset, config['model_path'], best_score, epoch, config['model_type'])

    scores = test(args, config, split="test")
    print_scores(scores)



def eval_select(args, config, model, tokenizer, validate_dataset, test_dataset, model_path, best_score, epoch, arch):
    scores_dev, _ = test(args, config, split="validate", model=model, tokenizer=tokenizer, test_dataset=validate_dataset)
    print_scores(scores_dev, mode="dev")
    scores_test, prediction_list = test(args, config, split="test", model=model, tokenizer=tokenizer, test_dataset=test_dataset)
    print_scores(scores_test)

    metrics = [x.strip() for x in config['metrics'].split(',')]

    if scores_dev[metrics[0]] > best_score:
        best_score = scores_dev[metrics[0]]
        # Save pytorch-model
        print("Save PyTorch model to {}".format(model_path))

        if args.record_result:
            with open('data/submission/{}.tsv'.format(config['result_path']), 'w+') as f:
                f.write("index\tprediction\n")
                for index, pred in enumerate(prediction_list):
                    f.write("{}\t{}\n".format(index, pred))

        save_checkpoint(epoch, arch, model, tokenizer, scores_dev, model_path)

    return best_score


def print_scores(scores, mode="test"):
    print()
    print("[{}] ".format(mode), end="")
    for sn, score in scores.items():
        print("{}: {}".format(sn, score), end=" ")
    print()


def save_checkpoint(epoch, arch, model, tokenizer, scores, filename):
    state = {
        'epoch': epoch,
        'arch': arch,
        'model': model,
        'tokenizer': tokenizer, 
        'scores': scores
    }
    torch.save(state, filename)


def load_checkpoint(filename):
    print("Load PyTorch model from {}".format(filename))
    state = torch.load(filename)
    return state['epoch'], state['arch'], state['model'], state['tokenizer'], state['scores']


def test(args, config, split="test", model=None, tokenizer=None, test_dataset=None):
    if model is None:
        epoch, arch, model, tokenizer, scores = load_checkpoint(config['model_path'])
    if test_dataset is None: 
        print("Load test set")
        test_dataset = load_data(config['data_path'], config['dataset'], config[split], 
            int(config['batch_size']), tokenizer, args.device, label_type=config['label_type'])
    
    model.eval()
    prediction_score_list, prediction_index_list, labels = [], [], []
    f = open(args.output_path, "w")
    lineno = 1
    for batch in test_dataset:
        # predictions, _ = model.test_step(tokens_tensor, segments_tensor, mask_tensor)
        if config['model_type'] == "specific_shared":
            predictions = model.test_step(*batch[:3])
        elif config['model_type'] == "siamese_bert":
            predictions = model(*batch[:6])
        else:
            predictions = model(*batch[:3])

        predicted_index = get_predicted_index(predictions.cpu().detach().numpy())
        prediction_index_list += predicted_index

        predicted_score = get_predicted_score(predictions.cpu().detach().numpy())
        prediction_score_list.extend(predicted_score)

        labels.extend(list(batch[-1].cpu().detach().numpy()))
        for p in predicted_index:
            f.write("{}\t{}\n".format(lineno, p))
            lineno += 1
        del predictions
    
    f.close()

    pearsonr, spearmanr, acc, p1, prec, rec, f1 = 0, 0, 0, 0, 0, 0, 0
    metrics = [x.strip() for x in config['metrics'].split(',')]
    scores = {}
    # if split != "test":
    if 'pearsonr' in metrics: scores['pearsonr'] = get_pearsonr(prediction_score_list, labels)
    if 'spearmanr' in metrics: scores['spearmanr'] = get_spearmanr(prediction_score_list, labels)

    if 'acc' in metrics: scores['acc'] = get_acc(prediction_index_list, labels)
    if 'p@1' in metrics: scores['p@1'] = get_p1(prediction_score_list, labels, config['data_path'], config['dataset'], config[split])
    if 'matthewsr' in metrics: scores['matthewsr'] = matthews_corrcoef(labels, prediction_index_list)

    if 'precision' in metrics or "recall" in metrics or "f1" in metrics:
        pre, rec, f1 = get_pre_rec_f1(prediction_index_list, labels)
        scores['precision'] = pre
        scores['recall'] = rec
        scores['f1'] = f1

    if metrics[0] in ['pearsonr', 'spearmanr', 'p@1']:
        return_list = prediction_score_list
    elif metrics[0] in ['acc', 'precision', 'recall', 'f1', 'matthewsr']:
        return_list = prediction_index_list

    torch.cuda.empty_cache()
    model.train()
    
    return scores, return_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='[train, test]')
    parser.add_argument('--config')
    parser.add_argument('--device', default='cuda', help='[cuda, cpu]')
    parser.add_argument('--load_trained', action='store_true', default=False, help='')
    parser.add_argument('--record_result', action='store_true', default=False, help='determine if record preds')
    parser.add_argument('--eval_steps', default=2500, type=int, help='evaluation per [eval_steps] steps, -1 for evaluation per epoch')
    parser.add_argument('--output_path', default='prediction.tmp', help='')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    print_config(config)
    
    if args.mode == "train":
        print("Start training...")
        train(args, config['settings'])
    elif args.mode == "transfer":
        print("Start transferring...")
        transfer(args, config['settings'])
    else:
        scores = test(args)
        print_scores(scores)
