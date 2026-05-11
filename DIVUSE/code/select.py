# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import argparse
import glob
import logging
import os
import pdb
import pickle
import random
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm
import multiprocessing
from model import Model
import pandas as pd
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import auc
# model reasoning
from captum.attr import LayerIntegratedGradients, DeepLift, DeepLiftShap, GradientShap, Saliency
# word-level tokenizer
from tokenizers import Tokenizer

# from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import RandomOverSampler

import DIVUSE
import generate_importance_score as gis
import time

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 i,
                 project):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.index = i
        self.project = project


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train", dataset="none"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "part":
            file_path = args.part_data_file
        else:
            file_path = ""
        self.examples = []
        if "holdout" in os.path.basename(file_path):
            df["split"].replace("holdout", "test")

        if file_path.endswith(".jsonl"):
            if not os.path.exists(file_path):
                print(file_path)
                os.makedirs(file_path)
                assert 0
            df = pd.read_json(file_path, orient="records", lines=True)
        elif file_path.endswith(".csv"):
            if not os.path.exists(file_path):
                print(file_path)
                assert 0
            df = pd.read_csv(file_path, index_col=0)
        else:
            raise NotImplementedError(file_path)

        if args.sample:
            df = df.sample(100)

        if "processed_func" in df.columns:
            func_key = "processed_func"
        elif "func" in df.columns:
            func_key = "func"
        funcs = df[func_key].tolist()
        labels = df["target"].astype(int).tolist()
        if not file_path.endswith(".csv"):
            indices = df["idx"].astype(int).tolist()
        else:
            indices = list(range(len(df)))

        project = []
        if dataset != "none":
            project = df["file_name"].tolist()
            file_names = []
            for i in tqdm(range(len(project))):
                project[i] = project[i].split("_")[0]
                file_names.append(project[i])


        for i in tqdm(range(len(funcs)), desc=f"load {file_type} dataset"):
            if len(project) != 0:
                self.examples.append(
                    convert_examples_to_features(funcs[i], labels[i], tokenizer, args, indices[i], project[i]))
            else:
                self.examples.append(
                    convert_examples_to_features(funcs[i], labels[i], tokenizer, args, indices[i], dataset))
        self.return_index = args.eval_export


    def __len__(self):
        return len(self.examples)

    def set_return_index(self, return_index):
        self.return_index = return_index

    def __getitem__(self, i):
        if self.return_index:
            return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(
                self.examples[i].index)
        else:
            return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


def convert_examples_to_features(func, label, tokenizer, args, i, project="none"):
    if args.use_word_level_tokenizer:
        encoded = tokenizer.encode(func)
        encoded = encoded.ids
        if len(encoded) > 510:
            encoded = encoded[:510]
        encoded.insert(0, 0)
        encoded.append(2)
        if len(encoded) < 512:
            padding = 512 - len(encoded)
            for _ in range(padding):
                encoded.append(1)
        source_ids = encoded
        source_tokens = []
        return InputFeatures(source_tokens, source_ids, label, i, project)
    # source
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label, i, project)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)



def main():
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    # parser.add_argument("--eval_data_file", default=None, type=str,
    #                     help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--part_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_export", action='store_true',
                        help="Whether to save prediction output.")
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--no_cuda", action='store_true')

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ")
    parser.add_argument("--reasoning_method", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    parser.add_argument('--ratio', type=float, default=1.0,
                        help="partset ratio")
    parser.add_argument('--handlesets', type=str, default="part", help="handlesets")
    parser.add_argument('--under', type=float, default=0.6, help="under sampling ratio")
    parser.add_argument('--over', type=str, default="vulfilter", help="over sampling ratio")
    parser.add_argument('--idx', type=int, default=0,
                        help="idx")

    # RQ3 - line-level evaluation
    parser.add_argument('--top_k_constant', type=int, default=10,
                        help="Top-K Accuracy constant")
    # num of attention heads
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help="number of attention heads used in CodeBERT")
    # raw predictions
    parser.add_argument("--write_raw_preds", default=False, action='store_true',
                        help="Whether to write raw predictions on test data.")
    # word-level tokenizer
    parser.add_argument("--use_word_level_tokenizer", default=False, action='store_true',
                        help="Whether to use word-level tokenizer.")
    # bpe non-pretrained tokenizer
    parser.add_argument("--use_non_pretrained_tokenizer", default=False, action='store_true',
                        help="Whether to use non-pretrained bpe tokenizer.")
    args = parser.parse_args()
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )
    # Set seed
    args.seed = int(args.handlesets)
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    config.num_attention_heads = args.num_attention_heads
    if args.use_word_level_tokenizer:
        print('using wordlevel tokenizer!')
        tokenizer = Tokenizer.from_file('./word_level_tokenizer/wordlevel.json')
    elif args.use_non_pretrained_tokenizer:
        tokenizer = RobertaTokenizer(vocab_file="bpe_tokenizer/bpe_tokenizer-vocab.json",
                                     merges_file="bpe_tokenizer/bpe_tokenizer-merges.txt")
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    if args.use_non_pretrained_model:
        model = RobertaForSequenceClassification(config=config)
    else:
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config,
                                                                 ignore_mismatched_sizes=True)

    model = Model(model, config, tokenizer, args)
    logger.info("Training/evaluation parameters %s", args)


    # if args.part_data_file is not None:
    train_name = os.path.basename(os.path.dirname(args.train_data_file))
    part_name = os.path.basename(os.path.dirname(args.part_data_file))
    save_path = os.path.join("./storage", f"storage_{train_name}_{part_name}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"create {save_path}")

    train_dataset_file = os.path.join(save_path, 'train_dataset.pkl')
    part_dataset_file = os.path.join(save_path, 'part_dataset.pkl')
    if os.path.exists(os.path.join(save_path, train_dataset_file)):
        train_dataset = pickle.load(open(train_dataset_file, 'rb'))
    else:
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        pickle.dump(train_dataset, open(train_dataset_file, 'wb'))
    if os.path.exists(os.path.join(save_path, part_dataset_file)):
        part_dataset = pickle.load(open(part_dataset_file, 'rb'))
    else:
        part_dataset = TextDataset(tokenizer, args, file_type='part')
        pickle.dump(part_dataset, open(part_dataset_file, 'wb'))

    logger.info(f"train dataset size: {len(train_dataset.examples)}")

    if args.handlesets != "none":
        l = len(part_dataset.examples)
        logger.info(f"part dataset size: {l}")
        if "vgx_ori" in args.part_data_file:
            const_part_pos = 15039
        elif "vulgen_ori" in args.part_data_file:
            const_part_pos = 68051
        elif "vulscriber" in args.part_data_file:
            const_part_pos = 15000
        else:
            const_part_pos = l
        const_part_neg = 240624
        part_pos_num = 0
        part_neg_num = 0
        pos_examples = []
        neg_examples = []

        for i in range(len(part_dataset.examples)):
            if part_dataset.examples[i].label == 1:
                part_pos_num += 1
                pos_examples.append(part_dataset.examples[i])
            elif part_dataset.examples[i].label == 0:
                part_neg_num += 1
                neg_examples.append(part_dataset.examples[i])
        logger.info(f"part pos dataset size: {len(pos_examples)}")
        logger.info(f"part neg dataset size: {len(neg_examples)}")

        if args.under != 1.0:
            random.seed(int(args.handlesets))
            random.shuffle(pos_examples)
            random.shuffle(neg_examples)
            random.seed(args.seed)
        pos_examples = pos_examples[:const_part_pos]
        neg_examples = neg_examples[:const_part_neg]
        logger.info(f"part pos dataset size: {len(pos_examples)}")
        logger.info(f"part neg dataset size: {len(neg_examples)}")

        if args.under != 1.0:
            if args.over == "DIVUSE":
                output_dir = os.path.dirname(args.output_dir)
                part_dataset.examples = pos_examples
                DIVUSE.get_cls(args, model, train_dataset, part_dataset, save_path)
                # pdb.set_trace()
                train_pos, train_neg, part_pos, part_neg = DIVUSE.load_cls(save_path)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    print(f"create {save_path}")
                DIVUSE.selector(train_pos, train_neg, part_pos, part_neg, save_path, output_dir, args.handlesets,
                                   args.under, args.over, args)
                DIVUSE.filter(train_dataset, part_dataset, output_dir, args.handlesets)
                neg_examples = neg_examples[:int(len(neg_examples) * args.under)]
                logger.info(f"part neg dataset size: {len(neg_examples)}")
                train_dataset.examples += part_dataset.examples
                train_dataset.examples += neg_examples
            elif args.over == "random":
                random.seed(int(args.handlesets))
                random.shuffle(pos_examples)
                random.shuffle(neg_examples)
                random.seed(args.seed)
                pos_examples = pos_examples[:int(len(pos_examples) * args.under)]
                logger.info(f"part pos dataset size: {len(pos_examples)}")
                neg_examples = neg_examples[:int(len(neg_examples) * args.under)]
                logger.info(f"part neg dataset size: {len(neg_examples)}")
                train_dataset.examples += pos_examples
                train_dataset.examples += neg_examples
            else:
                output_dir = os.path.dirname(args.output_dir)
                codebert_model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path)
                part_dataset.examples = pos_examples
                gis.get_importance_score(args, codebert_model, part_dataset, save_path)
                gis.filter(part_dataset, args.over, args.under, True, output_dir, save_path)
                neg_examples = neg_examples[:int(len(neg_examples) * args.under)]
                logger.info(f"part neg dataset size: {len(neg_examples)}")
                train_dataset.examples += part_dataset.examples
                train_dataset.examples += neg_examples
        else:
            train_dataset.examples += pos_examples
            train_dataset.examples += neg_examples

        logger.info(f"train dataset size: {len(train_dataset.examples)}")




if __name__ == "__main__":
    main()


