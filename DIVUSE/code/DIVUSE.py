from __future__ import absolute_import, division, print_function
import argparse
import glob
import json
import logging
import os
import pickle
import random
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm
import multiprocessing

import pandas as pd
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import auc
# model reasoning
from captum.attr import LayerIntegratedGradients, DeepLift, DeepLiftShap, GradientShap, Saliency
# word-level tokenizer
from tokenizers import Tokenizer
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
import pdb
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import defaultdict
import copy


def save_cls(args, model, dataloader, pos_path, neg_path):
    pos = []
    neg = []
    progress_bar = tqdm(dataloader, total=len(dataloader), desc="save cls")
    for mini_batch in progress_bar:
        (input_ids, labels, index) = mini_batch
        input_ids = input_ids.to(args.device)
        # print(input_ids)
        with torch.no_grad():
            cls_feature = model(input_ids=input_ids, cls_feature=True)
            if labels.item() == 1:
                pos.append((cls_feature[0].cpu().numpy(), index.item()))
            elif labels.item() == 0:
                neg.append((cls_feature[0].cpu().numpy(), index.item()))
    np.save(pos_path, np.array(pos, dtype=object))
    np.save(neg_path, np.array(neg, dtype=object))


def get_cls(args, model, train_dataset, part_dataset, save_path):
    train_pos_path = os.path.join(save_path, "train_pos.npy")
    train_neg_path = os.path.join(save_path, "train_neg.npy")
    part_pos_path = os.path.join(save_path, "part_pos.npy")
    part_neg_path = os.path.join(save_path, "part_neg.npy")
    if not os.path.exists(train_pos_path) or not os.path.exists(train_neg_path):
        train_dataset.set_return_index(True)
        train_sampler = SequentialSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1, num_workers=0)
        model.eval()
        model.to(args.device)
        save_cls(args, model, train_dataloader, train_pos_path, train_neg_path)
        train_dataset.set_return_index(False)
    if not os.path.exists(part_pos_path) or not os.path.exists(part_neg_path):
        part_dataset.set_return_index(True)
        part_sampler = SequentialSampler(part_dataset)
        part_dataloader = DataLoader(part_dataset, sampler=part_sampler, batch_size=1, num_workers=0)
        model.eval()
        model.to(args.device)
        save_cls(args, model, part_dataloader, part_pos_path, part_neg_path)
        part_dataset.set_return_index(False)


def load_cls(save_path):
    train_pos_path = os.path.join(save_path, "train_pos.npy")
    train_neg_path = os.path.join(save_path, "train_neg.npy")
    part_pos_path = os.path.join(save_path, "part_pos.npy")
    part_neg_path = os.path.join(save_path, "part_neg.npy")
    train_pos = np.load(train_pos_path, allow_pickle=True)
    train_neg = np.load(train_neg_path, allow_pickle=True)
    part_pos = np.load(part_pos_path, allow_pickle=True)
    part_neg = np.load(part_neg_path, allow_pickle=True)
    return train_pos, train_neg, part_pos, part_neg


def get_part_part_dist(part_cls, part_labels, part_indices, label, save_path, output_dir, k, args=None):
    if label == True:
        distances_path = os.path.join(save_path, "part_pos_knn_distances.npy")
    else:
        distances_path = os.path.join(save_path, "part_neg_knn_distances.npy")
    if os.path.exists(distances_path):
        distances = np.load(distances_path)
    else:

        k = min(len(part_cls), k)
        print("k:", k)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(part_cls, part_labels)
        distances = np.zeros(len(part_cls))
        for i, part_feature in tqdm(enumerate(part_cls), total=len(part_cls), desc="part-part knn"):
            distance, indices = knn.kneighbors([part_feature])
            distances[i] = np.mean(distance)
        np.save(distances_path, distances)
    # pdb.set_trace()
    return distances

def get_train_part_dist(train_cls, train_labels, part_cls, part_labels, part_indices, label, save_path, output_dir, k, args=None):
    if label == True:
        distances_path = os.path.join(save_path, "train_pos_knn_distances.npy")
    else:
        distances_path = os.path.join(save_path, "train_neg_knn_distances.npy")
    if os.path.exists(distances_path):
        distances = np.load(distances_path)
    else:
        if args is not None:
            k = args.KG
        k = min(len(train_cls), k)
        print("k:", k)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_cls, train_labels)
        distances = np.zeros(len(part_cls))
        for i, part_feature in tqdm(enumerate(part_cls), total=len(part_cls), desc="train-part knn"):
            distance, indices = knn.kneighbors([part_feature])
            distances[i] = np.mean(distance)
        np.save(distances_path, distances)
    # pdb.set_trace()
    return distances

def get_uncertain_score(part_part_dist, train_part_dist, part_indices, label, save_path, output_dir, args=None):
    alpha = 1.0
    beta = 1.0
    if args is not None:
        alpha = args.alpha
        beta = args.beta
    if label == True:
        rank_list_path = os.path.join(save_path, f"part_pos_rank_list_alpha{alpha}_beta{beta}.json")
    else:
        rank_list_path = os.path.join(save_path, f"part_neg_rank_list_alpha{alpha}_beta{beta}.json")
    if os.path.exists(rank_list_path):
    # if os.path.exists(rank_list_path) and False:
        with open(rank_list_path, 'r') as f:
            uncertain_score = json.load(f)
    else:
        intra_similarity = - part_part_dist
        inter_similarity = - train_part_dist
        intra_similarity = 1 / (intra_similarity - 1)
        inter_similarity = 1 / (inter_similarity - 1)
        uncertain_score = defaultdict(list)
        for i in range(len(part_part_dist)):
            uncertain_score[i].append(alpha * intra_similarity[i] + beta * inter_similarity[i])
        uncertain_score = sorted(uncertain_score.items(), key=lambda x: sum(x[1]), reverse=True)
        # part_part_dist = 1 / (part_part_dist + 1)
        # train_part_dist = 1 / (train_part_dist + 1)
        #
        # uncertain_score = defaultdict(list)
        # for i in range(len(part_part_dist)):
        #     uncertain_score[i].append(alpha * part_part_dist[i] + beta * train_part_dist[i])
        #
        # uncertain_score = sorted(uncertain_score.items(), key=lambda x: sum(x[1]))
        with open(rank_list_path, 'w') as f:
            json.dump(uncertain_score, f, indent=4)
    return uncertain_score



def selector_inclass(train_dict, part_dict, label, save_path, output_dir, k=100, args=None):
    normalizer = Normalizer(norm='l2')
    train_cls = np.array([i[0] for i in train_dict])
    train_cls = normalizer.transform(train_cls)
    train_indices = np.array([i[1] for i in train_dict])

    part_cls = np.array([i[0] for i in part_dict])
    part_cls = normalizer.transform(part_cls)
    part_indices = np.array([i[1] for i in part_dict])
    # pdb.set_trace()

    if label == True:
        train_labels = np.array([1] * len(train_cls))
        part_labels = np.array([1] * len(part_cls))
    else:
        train_labels = np.array([0] * len(train_cls))
        part_labels = np.array([0] * len(part_cls))

    part_part_dist = get_part_part_dist(part_cls, part_labels, part_indices, label, save_path, output_dir, k, args)
    train_part_dist = get_train_part_dist(train_cls, train_labels, part_cls, part_labels, part_indices, label, save_path, output_dir, k, args)
    uncertain_score = get_uncertain_score(part_part_dist, train_part_dist, part_indices, label, save_path, output_dir, args)
    return uncertain_score




def selector(train_pos, train_neg, part_pos, part_neg, save_path, output_dir, handlesets, under=0.6, over=0.0, args=None, over_rate=2, k=100, T=0.1):
    corr_indices_path = os.path.join(output_dir, "corr_indices.txt")
    if os.path.exists(corr_indices_path):
    # if os.path.exists(corr_indices_path) and False:
        corr_indices = np.loadtxt(corr_indices_path, dtype=int)
    else:
        score = defaultdict(list)
        pos_score = selector_inclass(train_pos, part_pos, True, save_path, output_dir, k, args)
        for k, v in pos_score:
            score[k].extend(v)
        if len(part_neg) != 0:
            neg_score = selector_inclass(train_neg, part_neg, False, save_path, output_dir, k, args)
            for k, v in neg_score:
                score[k].extend(v)
        score = sorted(score.items(), key=lambda x: sum(x[1]))
        indices = [i[0] for i in score]
        num = int((len(part_pos) + len(part_neg)) * under)
        print("num:", num, "indices:", len(indices))
        if num < len(indices):
            indices = indices[:num]
        else:
            assert 0
        corr_indices = sorted(indices)

        np.savetxt(corr_indices_path, corr_indices, fmt='%d')

    return corr_indices

def filter(train_dataset, part_dataset, output_dir, handlesets):
    corr_indices_path = os.path.join(output_dir, "corr_indices.txt")
    corr_indices = np.loadtxt(corr_indices_path, dtype=int)

    tmp_examples = []
    pos = 0
    neg = 0

    for i in corr_indices:
        for j in range(len(part_dataset.examples)):
            if part_dataset.examples[j].index == i:
                tmp_examples.append(part_dataset.examples[j])
                if part_dataset.examples[j].label == 1:
                    pos += 1
                elif part_dataset.examples[j].label == 0:
                    neg += 1

    print("part_dataset.examples:", len(part_dataset.examples), "pos:", pos, "neg:", neg)
    part_dataset.examples = tmp_examples
    print("part_dataset.examples:", len(part_dataset.examples))


if __name__ == "__main__":
    pass