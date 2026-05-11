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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm
import multiprocessing

from datasets import load_metric
import pandas as pd
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import auc
# model reasoning
from captum.attr import LayerIntegratedGradients, DeepLift, DeepLiftShap, GradientShap, Saliency
# word-level tokenizer
from tokenizers import Tokenizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata
import umap.umap_ as umap
import matplotlib
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.cluster import DBSCAN
import pdb
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import hdbscan
import seaborn as sns
from collections import defaultdict
import copy
from accelerate import Accelerator
from kmeans_pytorch import kmeans
matplotlib.use('Agg')

"""Calculate td metrics"""
def EL2N(td_log, dataset, data_importance, num_labels, max_epoch=10):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        targets.append(dataset.examples[i].label)

    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)
    data_importance['el2n'] = torch.zeros(data_size).type(torch.float32)
    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_labels)
        el2n_score = torch.sqrt(l2_loss(label_onehot,output).sum(dim=1))

        data_importance['el2n'][index] += el2n_score

    for i, item in enumerate(td_log):
        record_training_dynamics(item)

"""Calculate td metrics"""
def training_dynamics_metrics(td_log, dataset, data_importance):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        targets.append(dataset.examples[i].label)

    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)

    data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
    data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['accumulated_margin'] = torch.zeros(data_size).type(torch.float32)
    data_importance['variance'] = []
    for i in range(data_size):
        data_importance['variance'].append([])

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        correctness = (predicted == label).type(torch.int)
        # pdb.set_trace()
        data_importance['forgetting'][index] += torch.logical_and(data_importance['last_correctness'][index] == 1, correctness == 0)
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += data_importance['last_correctness'][index]

        batch_idx = range(output.shape[0])
        target_prob = output[batch_idx, label]
        for i, idx in enumerate(index):
            data_importance['variance'][idx].append(target_prob[i].item())
        output[batch_idx, label] = 0
        other_highest_prob = torch.max(output, dim=1)[0]
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin

    for i, item in enumerate(td_log):
        record_training_dynamics(item)

    # compute variance
    sizes = [len(data_importance['variance'][i]) for i in range(data_size)]
    for i, s in enumerate(sizes):
        if s != sizes[0]:
            for j in range(sizes[0] - s):
                data_importance['variance'][i].append(1.0)
    data_importance['variance'] = torch.tensor(np.std(np.array(data_importance['variance']), axis=-1))
    # data_importance['variance'] = data_importance['variance'] - np.tile(np.expand_dims(np.mean(data_importance['variance'], axis=-1), axis=1), (1, sizes[0]))


def save_importance_score(args, model, dataset, dataloader, basic_path, dynamics_path, advanced_path):
    if not os.path.exists(basic_path) or not os.path.exists(dynamics_path):
        deepginis = []
        confidences = []
        losses = []
        entropies = []
        training_dynamics = []
        progress_bar = tqdm(dataloader, total=len(dataloader), desc="save importance score")
        for idx, mini_batch in enumerate(progress_bar):
            (input_ids, labels, index) = mini_batch
            input_ids = input_ids.to(args.device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=labels)
                probs = F.softmax(outputs.logits.detach().cpu(), dim=-1)
                # deepgini = 1 - torch.sum(probs ** 2, dim=1)
                # deepginis.extend(deepgini.detach().cpu().numpy().tolist())
                # pdb.set_trace()
                confs = [probs[k, l].item() for k, l in enumerate(labels)]
                entropy = -1 * probs * torch.log(probs + 1e-10)
                entropy = torch.sum(entropy, dim=1).detach().cpu().numpy().tolist()
                entropies.extend(entropy)
                loss = F.cross_entropy(outputs.logits.detach().cpu(), labels, reduction='none').detach().cpu().numpy().tolist()
                losses.extend(loss)
                # losses.extend([outputs.loss.detach().cpu().numpy().tolist()])
                confidences.extend(confs)
                log_tuple = {
                    'idx': torch.tensor(idx).type(torch.long).unsqueeze(0),
                    # 对数概率
                    'output': F.log_softmax(outputs.logits.detach().cpu(), dim=1).detach().cpu().type(torch.half)
                }
                training_dynamics.append(log_tuple)
                # pdb.set_trace()
        data_importance = {}
        assert len(confidences) == len(entropies)
        assert len(entropies) == len(losses)
        data_importance['entropy'] = torch.zeros(len(entropies))
        data_importance['loss'] = torch.zeros(len(entropies))
        data_importance['confidence'] = torch.zeros(len(entropies))
        for i in range(len(confidences)):
            data_importance['entropy'][i] = entropies[i]
            data_importance['loss'][i] = losses[i]
            data_importance['confidence'][i] = confidences[i]
        with open(basic_path, 'wb') as f:
            pickle.dump(data_importance, f)
        with open(dynamics_path, 'wb') as f:
            pickle.dump(training_dynamics, f)

    if not os.path.exists(advanced_path):
        with open(basic_path, 'rb') as f:
            data_importance = pickle.load(f)
        with open(dynamics_path, 'rb') as f:
            training_dynamics = pickle.load(f)
        training_dynamics_metrics(training_dynamics, dataset, data_importance)
        EL2N(training_dynamics, dataset, data_importance, 2, max_epoch=0)
        with open(advanced_path, 'wb') as f:
            pickle.dump(data_importance, f)

def get_importance_score(args, model, part_dataset, save_path):
    part_basic_score_path = os.path.join(save_path, "part_basic_score.pickle")
    part_training_dynamics_path = os.path.join(save_path, "part_training_dynamics.pickle")
    part_advanced_score_path = os.path.join(save_path, "part_advanced_score.pickle")
    if not os.path.exists(part_advanced_score_path):
        part_dataset.set_return_index(True)
        part_sampler = SequentialSampler(part_dataset)
        part_dataloader = DataLoader(part_dataset, sampler=part_sampler, batch_size=1, num_workers=0)
        model.eval()
        model.to(args.device)
        save_importance_score(args, model, part_dataset, part_dataloader, part_basic_score_path, part_training_dynamics_path, part_advanced_score_path)
        part_dataset.set_return_index(False)
    with open(part_advanced_score_path, 'rb') as f:
        part_advanced_score = pickle.load(f)
    # pdb.set_trace()
    return part_advanced_score

def get_median(features, targets):
    # get the median feature vector of each class
    num_classes = len(np.unique(targets, axis=0))
    prot = np.zeros((num_classes, features.shape[-1]), dtype=features.dtype)
    for i in range(num_classes):
        # prot[i] = np.median(features[(targets == i).nonzero(), :].squeeze(), axis=0, keepdims=False)
        prot[i] = np.mean(features[(targets == 1).nonzero(), :].squeeze(), axis=0, keepdims=False)
    return prot

def get_distance(features, labels):
    prots = get_median(features, labels)
    prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))
    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        # prots_for_each_example[(labels == i).nonzero()[0], :] = prots[i]
        prots_for_each_example[(labels == 1).nonzero()[0], :] = prots[i]
    distance = np.linalg.norm(features - prots_for_each_example, axis=1)
    return distance

def moderate(data_score, ratio, features):
    def get_prune_idx(rate, distance):
        rate = 1 - rate
        low = 0.5 - rate / 2
        high = 0.5 + rate / 2
        sorted_idx = distance.argsort()
        low_idx = round(distance.shape[0] * low)
        high_idx = round(distance.shape[0] * high)
        ids = np.concatenate((sorted_idx[:low_idx], sorted_idx[high_idx:]))
        return ids
    targets_list = data_score['targets']
    distance = get_distance(features, targets_list)
    ids = get_prune_idx(ratio, distance)
    return ids

def find_desired_samples(reps, cluster_ids_x, cluster_centers):
    distance_list = []
    for i in tqdm(range(reps.shape[0])):
        rep = reps[i]
        cluster_id = cluster_ids_x[i]
        cluster_center = cluster_centers[cluster_id]
        distance = torch.linalg.norm(rep - cluster_center, dim=0).item()
        distance_list.append(distance)
    return distance_list


def selfsupproto(ratio, features):
    features = torch.tensor(features)
    cluster_ids_x, cluster_centers = kmeans(X=features,
                                            num_clusters=10,
                                            distance='euclidean',
                                            tol=1e-5)
    # pdb.set_trace()
    distance_list = find_desired_samples(reps=features,
                                         cluster_ids_x=cluster_ids_x,
                                         cluster_centers=cluster_centers)
    distance_tensor = torch.tensor(distance_list)
    _, top_indices = torch.topk(distance_tensor, int(ratio * len(distance_tensor)), largest=True)
    # pdb.set_trace()
    return top_indices.tolist()

def filter(part_dataset, key, ratio, descending, output_dir, save_path):
    # descending == True: 从大到小
    part_advanced_score_path = os.path.join(save_path, "part_advanced_score.pickle")
    with open(part_advanced_score_path, 'rb') as f:
        part_advanced_score = pickle.load(f)
    if key == "moderate":
        part_pos = np.load(os.path.join(save_path, "part_pos.npy"), allow_pickle=True)
        features = np.array([i[0] for i in part_pos])
        coreset_index = moderate(part_advanced_score, ratio, features)
        coreset_index = np.sort(coreset_index)
    elif key == "selfsupproto":
        part_pos = np.load(os.path.join(save_path, "part_pos.npy"), allow_pickle=True)
        features = np.array([i[0] for i in part_pos])
        coreset_index = selfsupproto(ratio, features)
        coreset_index = np.sort(coreset_index)
        # pdb.set_trace()
    else:
        if key == "deepgini":
            # 1 - torch.sum(probs ** 2, dim=1)
            prob0 = part_advanced_score['confidence']
            prob1 = 1 - prob0
            score = 1 - prob0 ** 2 - prob1 ** 2
        else:
            assert key in part_advanced_score
            score = part_advanced_score[key]
        # pdb.set_trace()
        score_sorted_index = torch.argsort(score, descending=descending)
        # total_num = ratio * part_advanced_score[key].shape[0]
        total_num = ratio * len(score)
        print("Selecting from %s samples" % total_num)
        coreset_index = score_sorted_index[:int(total_num)]
        coreset_index, _ = torch.sort(coreset_index)
    np.savetxt(os.path.join(output_dir, "coreset_index.txt"), coreset_index, fmt='%d')

    tmp_examples = []
    pos = 0
    neg = 0
    for i in coreset_index:
        tmp_examples.append(part_dataset.examples[i])
        if part_dataset.examples[i].label == 1:
            pos += 1
        else:
            neg += 1

    # pdb.set_trace()
    print("part_dataset.examples:", len(part_dataset.examples), "pos:", pos, "neg:", neg)
    part_dataset.examples = tmp_examples
    print("part_dataset.examples:", len(part_dataset.examples))


def plot_score_distribution(scores, coreset_index, out_file, n_bins=25, coreset_key='forgetting', args=None):

    total_num = len(scores)
    coreset_num = len(coreset_index)

    score_min = np.min(scores)
    score_max = np.max(scores)

    if coreset_key == 'accumulated_margin':
        scores = -(scores - score_max)
        score_min = np.min(scores)
        score_max = np.max(scores)

    bins = np.linspace(score_min, score_max, n_bins + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3,4))
    counts = [np.sum((scores >= bins[i-1]) & (scores < bins[i]))/total_num for i in range(1, n_bins+1)]
    coreset_scores = scores[coreset_index]
    coreset_counts = [np.sum((coreset_scores >= bins[i-1]) & (coreset_scores < bins[i]))/coreset_num for i in range(1, n_bins+1)]
    coreset_counts = [max(1e-6, c) for c in coreset_counts]
    entropy = -(np.array(coreset_counts) * np.log(np.abs(np.array(coreset_counts)))).sum()
    print('Rendering with min score %s and max score %s' % (score_min, score_max))
    print(bins)
    print(coreset_counts)
    if coreset_key in ['forgetting']:
        ax1.bar(bins[1:], counts)
        ax2.bar(bins[1:], coreset_counts)
    else:
        ax1.bar(bins[1:], counts)
        ax2.bar(bins[1:], coreset_counts)
        # ax1.plot(bins[1:], counts)
        # ax2.plot(bins[1:], coreset_counts)
    ax1.set_ylim(0, max(counts) + 0.1)
    ax2.set_ylim(0, max(counts) + 0.1)
    ax2.set_xlabel('Difficulty Score', fontsize=10)
    ax1.set_ylabel('Fraction of Dataset', fontsize=10)
    ax2.set_ylabel('Fraction of Coreset', fontsize=10)
    ax1.set_title('Full dataset distribution', fontsize=10)
    ax2.set_title(r'$k$=%s, $\gamma_{r}$=%s, Entropy=%s' % (args.n_neighbor, round(args.gamma, 1), round(entropy, 3)), fontsize=10)
    #ax2.set_title(r'$k$=%s, $\gamma_{r}$=%s, Entropy=%s' % (100, 0.1, round(entropy, 3)), fontsize=10)
    ax1.grid(True, linestyle='--')
    ax2.grid(True, linestyle='--')
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    return entropy








