#!/bin/usr/env python3
# -*- coding: utf-8 -*-
# -------------------------------
# Author: SuphxLin
# CreateTime: 2021/03/26 12:18
# FileName: main.py
# Description: 
# Question:

import argparse
import numpy as np
import os

from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddle.vision.models import resnet50

import src.datasets.mvtec as mvtec


def parse_args():
    parser = argparse.ArgumentParser('SPADE')
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="./result")
    return parser.parse_args()


def main():
    args = parse_args()

    # device setup
    paddle.set_device("gpu")

    # load model
    model = resnet50(pretrained=True)
    model.eval()

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_post_hook(hook)
    model.layer2[-1].register_forward_post_hook(hook)
    model.layer3[-1].register_forward_post_hook(hook)
    model.avgpool.register_forward_post_hook(hook)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in mvtec.CLASS_NAMES:
        train_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32)
        test_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

        # extract train set features
        for (x, y, mask) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            with paddle.no_grad():
                pred = model(x)
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v)
            # initialize hook outputs
            outputs = []
        for k, v in train_outputs.items():
            train_outputs[k] = paddle.concat(v, 0)

        gt_list = []
        gt_mask_list = []
        test_imgs = []

        # extract test set features
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with paddle.no_grad():
                pred = model(x)
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v)
            # initialize hook outputs
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = paddle.concat(v, 0)

        # calculate distance matrix
        dist_matrix = calc_dist_matrix(paddle.flatten(test_outputs['avgpool'], 1),
                                       paddle.flatten(train_outputs['avgpool'], 1))

        # select K nearest neighbor and take average
        topk_values, topk_indexes = paddle.topk(dist_matrix, k=args.top_k, largest=False)
        scores = paddle.mean(topk_values, 1).cpu().detach().numpy()
        # calculate image-level ROC AUC score
        fpr, tpr, _ = roc_curve(gt_list, scores)
        roc_auc = roc_auc_score(gt_list, scores)
        total_roc_auc.append(roc_auc)
        print('%s ROCAUC: %.3f' % (class_name, roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))

        score_map_list = []
        for t_idx in tqdm(range(test_outputs['avgpool'].shape[0]), '| localization | test | %s |' % class_name):
            score_maps = []
            for layer_name in ['layer1', 'layer2', 'layer3']:  # for each layer

                # construct a gallery of features at all pixel locations of the K nearest neighbors
                topk_feat_map = paddle.stack(
                    [train_outputs[layer_name][idx] for idx in topk_indexes[t_idx].numpy().tolist()])

                test_feat_map = test_outputs[layer_name][t_idx:t_idx + 1]
                feat_gallery = paddle.transpose(topk_feat_map, [0, 3, 2, 1])
                feat_gallery = paddle.flatten(feat_gallery, start_axis=0, stop_axis=2).unsqueeze(-1).unsqueeze(-1)
                # calculate distance matrix
                dist_matrix_list = []
                for d_idx in range(feat_gallery.shape[0] // 100):
                    dist = paddle.nn.PairwiseDistance()
                    dist_matrix = dist(feat_gallery[d_idx * 100:d_idx * 100 + 100], test_feat_map)
                    dist_matrix_list.append(dist_matrix)
                dist_matrix = paddle.concat(dist_matrix_list, 0)

                # k nearest features from the gallery (k=1)
                score_map = paddle.min(dist_matrix, axis=0)
                score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=[224, 224], mode='bilinear',
                                          align_corners=False)

                score_maps.append(score_map)

            # average distance between the features
            score_map = paddle.mean(paddle.concat(score_maps, 0), axis=0)

            # apply gaussian smoothing on the score map
            score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
            score_map_list.append(score_map)

        flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel().astype(int)
        flatten_score_map_list = np.concatenate(score_map_list).ravel()
        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
        per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('%s pixel ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))

        # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list, flatten_score_map_list)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.nanargmax(f1)]

        # visualize localization result
        visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, args.save_path, class_name, vis_num=5)

    print('Average ROCAUC: %.3f' % np.nanmean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.nanmean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.nanmean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.nanmean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def calc_dist_matrix(x, y):
    """Calculate Euclidean distance matrix with paddle.Tensor"""
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    x = x.unsqueeze(1)
    x = paddle.expand(x, [n, m, d])
    y = y.unsqueeze(0)
    y = paddle.expand(y, [n, m, d])
    dist_matrix = paddle.sqrt(paddle.pow(x - y, 2).sum(2))
    return dist_matrix


def visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold,
                         save_path, class_name, vis_num=5):
    for t_idx in range(vis_num):
        test_img = test_imgs[t_idx]
        test_img = denormalization(test_img)
        test_gt = gt_mask_list[t_idx].transpose(1, 2, 0).squeeze()
        test_pred = score_map_list[t_idx]
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
        test_pred_img = test_img.copy()
        test_pred_img[test_pred == 0] = 0

        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 4))
        fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(test_img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(test_gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(test_pred, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')
        ax_img[3].imshow(test_pred_img)
        ax_img[3].title.set_text('Predicted anomalous image')

        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        fig_img.savefig(os.path.join(save_path, 'images', '%s_%03d.png' % (class_name, t_idx)), dpi=100)
        fig_img.clf()
        plt.close(fig_img)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


if __name__ == '__main__':
    main()
