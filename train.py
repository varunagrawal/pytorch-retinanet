import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from datasets import CocoDataset, YCBDataset, CSVDataset, collater, Resizer, \
    AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):

    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')

    parser.add_argument(
        '--dataset', help='Dataset type, must be one of csv or coco or ycb.')
    parser.add_argument('--path', help='Path to dataset directory')
    parser.add_argument(
        '--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument(
        '--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument(
        '--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument(
        '--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs',
                        type=int, default=100)
    parser.add_argument('--evaluate_every', default=20, type=int)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.path is None:
            raise ValueError(
                'Must provide --path when training on non-CSV datasets')

        dataset_train = CocoDataset(parser.path,
                                    ann_file='instances_train2014.json',
                                    set_name="train2014",
                                    transform=transforms.Compose([Normalizer(), Augmenter(),
                                                                  Resizer(min_side=512,
                                                                          max_side=512)]))
        dataset_val = CocoDataset(parser.path, ann_file='instances_val2014.cars.json',
                                  set_name="val2014",
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'ycb':

        dataset_train = YCBDataset(parser.path,
                                   "image_sets/train.txt",
                                   transform=transforms.Compose([Normalizer(),
                                                                 Augmenter(),
                                                                 Resizer(min_side=512,
                                                                         max_side=512)]),
                                   train=True)
        dataset_val = YCBDataset(parser.path, "image_sets/val.txt",
                                 transform=transforms.Compose(
                                     [Normalizer(), Resizer()]),
                                 train=False)

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError(
                'Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(),
                                                                 Augmenter(),
                                                                 Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError(
            'Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(
        dataset_train, batch_size=12, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=8,
                                  collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(
            dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=4,
                                    collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(
            num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError(
            'Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    # retinanet = torch.nn.DataParallel(retinanet).cuda()

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                classification_loss, regression_loss = retinanet(
                    [data['img'].cuda().float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss.item()))
                epoch_loss.append(float(loss.item()))

                print('Epoch: {} | Iteration: {}/{} | "\
                    "Classification loss: {:1.5f} | "\
                    "Regression loss: {:1.5f} | "\
                        "Running loss: {:1.5f}'.format(epoch_num,
                                                       iter_num,
                                                       len(dataloader_train),
                                                       float(
                                                           classification_loss),
                                                       float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if ((epoch_num+1) % parser.evaluate_every == 0) or epoch_num+1 == parser.epochs:

            if parser.dataset == 'coco':

                print('Evaluating dataset')

                coco_eval.evaluate_coco(dataset_val, retinanet)

            elif parser.dataset == 'csv' and parser.csv_val is not None:

                print('Evaluating dataset')

                mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        # torch.save(retinanet, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()

    torch.save(retinanet, "retinanet_model_final.pt")


if __name__ == '__main__':
    main()
