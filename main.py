import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils import get_data
from src.model import GCN_CAPS_Model
from src.L2Regularization import Regularization
from src.eval_metrics import *


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch, weight_decay, reg_loss, args):
    results = []
    truths = []
    model.train()
    total_loss = 0.0
    total_batch_size = 0

    for ind, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
        # measure data loading time
        sample_ind, text, audio, video = batch_X
        text, audio, video = text.cuda(non_blocking=True), audio.cuda(non_blocking=True), video.cuda(non_blocking=True)
        batch_Y = batch_Y.cuda(non_blocking=True)
        eval_attr = batch_Y.squeeze(-1)
        batch_size = text.size(0)
        total_batch_size += batch_size

        preds = model(text, audio, video, batch_size)
        if args.dataset in ['mosi', 'mosei_senti']:
            preds = preds.reshape(-1)
            eval_attr = eval_attr.reshape(-1)
            raw_loss = criterion(preds, eval_attr)
            if weight_decay > 0:
                raw_loss = raw_loss + reg_loss(model)
            results.append(preds)
            truths.append(eval_attr)

        total_loss += raw_loss.item() * batch_size
        combined_loss = raw_loss
        optimizer.zero_grad()
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

    avg_loss = total_loss / total_batch_size
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


def validate(loader, model, criterion, args):
    model.eval()
    results = []
    truths = []
    total_loss = 0.0
    total_batch_size = 0
    with torch.no_grad():
        for ind, (batch_X, batch_Y, batch_META) in enumerate(loader):
            sample_ind, text, audio, video = batch_X
            text, audio, video = text.cuda(non_blocking=True), audio.cuda(non_blocking=True), video.cuda(non_blocking=True)
            batch_Y = batch_Y.cuda(non_blocking=True)
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            batch_size = text.size(0)
            total_batch_size += batch_size
            preds = model(text, audio, video, batch_size)
            if args.dataset in ['mosi', 'mosei_senti']:
                preds = preds.reshape(-1)
                eval_attr = eval_attr.reshape(-1)
                raw_loss = criterion(preds, eval_attr)
                results.append(preds)
                truths.append(eval_attr)
                total_loss += raw_loss.item() * batch_size

    avg_loss = total_loss / total_batch_size
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch GCN_CAPS Learner')
    parser.add_argument('--aligned', action='store_true', default=False, help='consider aligned experiment or not')
    parser.add_argument('--dataset', type=str, default='mosei_senti', help='dataset to use')
    parser.add_argument('--data-path', type=str, default='data', help='path for storing the dataset')
    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                        help='initial learning rate', dest='lr')
    parser.add_argument('--MULT_d', default=30, type=int, help='the output dimensionality of MULT is 2*MULT_d')
    parser.add_argument('--vertex_num', default=20, type=int, help='number of vertexes')
    parser.add_argument('--dim_capsule', default=32, type=int, help='dimension of capsule')
    parser.add_argument('--routing', default=3, type=int, help='total routing rounds')
    parser.add_argument('--weight_decay', default=0.001, type=float, help='L2Regularization')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout in primary capsule in StoG')
    parser.add_argument('--optimizer', default='RMSprop', type=str)
    parser.add_argument('--clip', type=float, default=1, help='gradient clip value (default: 1)')
    parser.add_argument('--patience', default=10, type=int, help='patience for learning rate decay')
    args = parser.parse_args()

    assert args.dataset in ['mosi', 'mosei_senti'], "supported datasets are mosei_senti and mosi"

    hyp_params = args
    hyp_params.MULT_d = args.MULT_d
    hyp_params.vertex_num = args.vertex_num
    hyp_params.dim_capsule = args.dim_capsule
    hyp_params.routing = args.routing
    hyp_params.weight_decay = args.weight_decay
    hyp_params.dropout = hyp_params.dropout
    current_setting = (hyp_params.MULT_d, hyp_params.vertex_num, hyp_params.dim_capsule, hyp_params.routing,
                       hyp_params.dropout, hyp_params.weight_decay,
                       args.optimizer, args.batch_size)

    if args.dataset == "mosi":
        criterion = nn.L1Loss().cuda()
        t_in = 300
        a_in = 5
        v_in = 20
        label_dim = 1
        if args.aligned:
            T_t = T_a = T_v = 50
        else:
            T_t = 50
            T_a = 375
            T_v = 500

    elif args.dataset == "mosei_senti":
        criterion = nn.L1Loss().cuda()
        t_in = 300
        a_in = 74
        v_in = 35
        label_dim = 1
        if args.aligned:
            T_t = T_a = T_v = 50
        else:
            T_t, T_a, T_v = 50, 500, 500

    model = GCN_CAPS_Model(args, label_dim, t_in, a_in, v_in, T_t, T_a, T_v,
                           hyp_params.MULT_d,
                           hyp_params.vertex_num,
                           hyp_params.dim_capsule,
                           hyp_params.routing,
                           hyp_params.dropout).cuda()

    weight_decay = args.weight_decay
    if weight_decay > 0:
        reg_loss = Regularization(model, weight_decay, p=2).cuda()
    else:
        reg_loss = 0

    if args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, factor=0.1, verbose=True)

    train_data = get_data(args, args.dataset, 'train')
    valid_data = get_data(args, args.dataset, 'valid')
    test_data = get_data(args, args.dataset, 'test')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    if args.dataset in ['mosi', 'mosei_senti']:
        best_acc = -1
        mae_best_acc = 2
        mult_a7_best_acc = -1
        mult_a5_best_acc = -1
        corr_best_acc = 0
        fscore_best_acc = 0
        patience_acc = 0

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train_loss, train_results, train_truth = train(train_loader, model, criterion, optimizer, epoch,
                                                       weight_decay, reg_loss, args)
        # validate for one epoch
        valid_loss, valid_results, valid_truth = validate(valid_loader, model, criterion, args)
        # test for one epoch
        test_loss, test_results, test_truth = validate(test_loader, model, criterion, args)
        scheduler.step(valid_loss)

        if args.dataset == "mosi":
            mae, corr, mult_a7, mult_a5, f_score, acc = eval_mosi(test_results, test_truth)
        elif args.dataset == 'mosei_senti':
            mae, corr, mult_a7, mult_a5, f_score, acc = eval_mosei_senti(test_results, test_truth)

        if args.dataset in ['mosi', 'mosei_senti']:
            print('Epoch {:2d} Loss| Train Loss{:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f}'
                  .format(epoch, train_loss, valid_loss, test_loss))
            if best_acc < acc:
                if args.aligned:
                    print('aligned {} dataset | acc improved! saving model to aligned_{}_best_model.pkl'
                          .format(args.dataset, args.dataset))
                    torch.save(model, 'aligned_{}_best_model.pkl'.format(args.dataset))
                else:
                    print('unaligned {} dataset | acc improved! saving model to unaligned_{}_best_model.pkl'
                          .format(args.dataset, args.dataset))
                    torch.save(model, 'unaligned_{}_best_model.pkl'.format(args.dataset))
                best_acc = acc
                mae_best_acc = mae
                mult_a7_best_acc = mult_a7
                mult_a5_best_acc = mult_a5
                corr_best_acc = corr
                fscore_best_acc = f_score
                patience_acc = 0
            else:
                patience_acc += 1
            # if patience_acc > 100:
            #     break

    if args.dataset in ['mosi', 'mosei_senti']:
        print("hyper-parameters: MULT_d, vertex_num, dim_capsule, routing, dropout, weight_decay,"
              "optimizer, batch_size", current_setting)
        print("Best Acc: {:5.4f}".format(best_acc))
        print("mae: {:5.4f}".format(mae_best_acc))
        print("mult_a7: {:5.4f}".format(mult_a7_best_acc))
        print("mult_a5: {:5.4f}".format(mult_a5_best_acc))
        print("corr: {:5.4f}".format(corr_best_acc))
        print("fscore: {:5.4f}".format(fscore_best_acc))
        print('-' * 50)

