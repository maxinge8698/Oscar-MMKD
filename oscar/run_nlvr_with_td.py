# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import argparse
import copy
import glob
import json
import logging
import os
import sys
import time

import _pickle as cPickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

sys.path.insert(0, '.')

from transformers.pytorch_transformers import BertConfig, BertTokenizer
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from transformers.pytorch_transformers import WEIGHTS_NAME

from oscar.modeling.modeling_bert import ImageBertForMultipleChoice, ImageBertForSequenceClassification
from oscar.utils.misc import set_seed
from oscar.utils.task_utils import _truncate_seq_pair, output_modes, processors

import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, ImageBertForSequenceClassification, BertTokenizer),
}

log_json = []

debug_size = 500


class NLVRDataset(Dataset):
    """ NLVR2 Dataset """

    def __init__(self, args, name, tokenizer, img_features):
        super(NLVRDataset, self).__init__()

        assert name in ['train', 'val', 'test1', 'val+test1']

        self.args = args
        self.name = name
        self.tokenizer = tokenizer

        self.output_mode = output_modes[args.task_name]

        # load image features
        self.img_features = img_features

        self.examples, self.labels = _load_dataset(args, name)

        self.label_map = {label: i for i, label in enumerate(self.labels)}

        logger.info('%s Data Examples: %d' % (name, len(self.examples)))

    def tensorize_example(self,
                          example,
                          cls_token_at_end=False,
                          pad_on_left=False,
                          cls_token='[CLS]',
                          sep_token='[SEP]',
                          cls_token_segment_id=1,
                          pad_token_segment_id=0,
                          pad_token=0,
                          sequence_a_segment_id=0,
                          sequence_b_segment_id=1,
                          mask_padding_with_zero=True):
        tokens_a = self.tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            text_b = example.text_b['left'] + ' ' + example.text_b['right']
            tokens_b = self.tokenizer.tokenize(text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
            _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)  # Account for [CLS], [SEP], [SEP] with "- 3"
        else:
            if len(tokens_a) > self.args.max_seq_length - 2:
                tokens_a = tokens_a[:(self.args.max_seq_length - 2)]  # Account for [CLS] and [SEP] with "- 2"

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length
        assert len(segment_ids) == self.args.max_seq_length

        # image features
        if self.args.img_feature_type.startswith('dis_code'):
            img_feat = self.img_features[example.img_key]
            if self.args.img_feature_type == 'dis_code_ln':  # for discrete code image representation
                img_feat = img_feat.reshape(-1, img_feat.shape[0])
            if self.args.img_feature_type == 'dis_code_t':  # transposed
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * 64
            else:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
        else:  # faster_r-cnn
            img_key_left = example.img_key['left']
            img_key_right = example.img_key['right']
            img_feat_left = self.img_features[img_key_left]
            img_feat_right = self.img_features[img_key_right]
            img_feat = torch.cat((img_feat_left, img_feat_right), 0)
            if img_feat.shape[0] > 2 * self.args.max_img_seq_length:
                img_feat = img_feat[0: 2 * self.args.max_img_seq_length, ]
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
            else:
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
                padding_matrix = torch.zeros((2 * self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
                img_feat = torch.cat((img_feat, padding_matrix), 0)
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                    # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]
        # print(len(input_ids), len(input_mask), len(segment_ids), img_feat.shape)

        if self.args.output_mode == "classification":
            if example.label is None:
                label_id = [0]
                # score = [0]
            else:
                # label_id = [self.label_map[l] for l in example.label]
                label_id = [example.label]
                # score = [0]
        elif self.args.output_mode == "regression":
            if len(example.label) == 0:
                label_id = 0
            else:
                label_id = float(example.label)
        else:
            raise KeyError(self.args.output_mode)

        if self.args.img_feature_type in ['dis_code', 'dis_code_t']:
            img_feat = img_feat.type(torch.long)
        elif self.args.img_feature_type in ['dis_code_ln']:
            # img_feat = img_feat.reshape(-1, img_feat.shape[0])
            img_feat = img_feat.type(torch.float)

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor([label_id[0]], dtype=torch.long),
                img_feat,
                torch.tensor([example.q_id], dtype=torch.long))

    def tensorize_example_pair(self,
                               example,
                               cls_token_at_end=False,
                               pad_on_left=False,
                               cls_token='[CLS]',
                               sep_token='[SEP]',
                               cls_token_segment_id=1,
                               pad_token_segment_id=0,
                               pad_token=0,
                               sequence_a_segment_id=0,
                               sequence_b_segment_id=1,
                               mask_padding_with_zero=True):
        tokens_a = self.tokenizer.tokenize(example.text_a)

        choices = []
        for choice_key in example.img_key:  # ("left", "right")
            tokens_b = None

            if example.text_b:
                tokens_b = self.tokenizer.tokenize(example.text_b[choice_key])
                # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
                _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)  # Account for [CLS], [SEP], [SEP] with "- 3"
            else:
                if len(tokens_a) > self.args.max_seq_length - 2:
                    tokens_a = tokens_a[:(self.args.max_seq_length - 2)]  # Account for [CLS] and [SEP] with "- 2"

            tokens = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.args.max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == self.args.max_seq_length
            assert len(input_mask) == self.args.max_seq_length
            assert len(segment_ids) == self.args.max_seq_length

            # img
            img_key = example.img_key[choice_key]
            img_feat = self.img_features[img_key]

            if img_feat.shape[0] > self.args.max_img_seq_length:
                img_feat = img_feat[0: self.args.max_img_seq_length, ]
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
            else:
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
                padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
                img_feat = torch.cat((img_feat, padding_matrix), 0)
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                    # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]
            # print(len(input_ids), len(input_mask), len(segment_ids), img_feat.shape)
            choices.append((tokens, input_ids, input_mask, segment_ids, img_feat))

        if self.args.output_mode == "classification":
            if example.label is None:
                label_id = [0]
                # score = [0]
            else:
                # label_id = [self.label_map[l] for l in example.label]
                label_id = [example.label]
                # score = [0]
        elif self.args.output_mode == "regression":
            if len(example.label) == 0:
                label_id = 0
            else:
                label_id = float(example.label)
        else:
            raise KeyError(self.args.output_mode)

        choice_input_ids = [choice[1] for choice in choices]
        choice_input_mask = [choice[2] for choice in choices]
        choice_input_segs = [choice[3] for choice in choices]
        choice_input_imgs = [choice[4] for choice in choices]

        choice_img_feats = torch.stack(choice_input_imgs, dim=0)

        return (torch.tensor(choice_input_ids, dtype=torch.long),
                torch.tensor(choice_input_mask, dtype=torch.long),
                torch.tensor(choice_input_segs, dtype=torch.long),
                torch.tensor(label_id, dtype=torch.long),
                choice_img_feats,
                torch.tensor([example.q_id], dtype=torch.long))

    def __getitem__(self, index):
        entry = self.examples[index]
        if self.args.use_pair:  # True
            example = self.tensorize_example_pair(entry,
                                                  cls_token_at_end=bool(self.args.model_type in ['xlnet']),
                                                  pad_on_left=bool(self.args.model_type in ['xlnet']),
                                                  cls_token=self.tokenizer.cls_token,
                                                  sep_token=self.tokenizer.sep_token,
                                                  cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
                                                  pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
        else:
            example = self.tensorize_example(entry,
                                             cls_token_at_end=bool(self.args.model_type in ['xlnet']),
                                             pad_on_left=bool(self.args.model_type in ['xlnet']),
                                             cls_token=self.tokenizer.cls_token,
                                             sep_token=self.tokenizer.sep_token,
                                             cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
                                             pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
        return example

    def __len__(self):
        return len(self.examples)


def _load_dataset(args, name):
    processor = processors[args.task_name]()
    labels = processor.get_labels()

    if name == 'train':
        examples = processor.get_train_examples(args.data_dir, args.use_label_seq, 'nlvr2_train.json')
    elif name == 'val':
        if args.eval_data_type == 'bal':
            examples = processor.get_dev_examples(args.data_dir, args.use_label_seq, 'nlvr2_balanced_dev.json')
        elif args.eval_data_type == 'unbal':
            examples = processor.get_dev_examples(args.data_dir, args.use_label_seq, 'nlvr2_unbalanced_dev.json')
        else:  # all
            examples = processor.get_dev_examples(args.data_dir, args.use_label_seq, 'nlvr2_dev.json')
    elif name == 'test1':
        if args.test_data_type == 'bal':
            examples = processor.get_test_examples(args.data_dir, args.use_label_seq, 'nlvr2_balanced_test1.json')
        elif args.test_data_type == 'unbal':
            examples = processor.get_test_examples(args.data_dir, args.use_label_seq, 'nlvr2_unbalanced_test1.json')
        else:  # all
            examples = processor.get_test_examples(args.data_dir, args.use_label_seq, 'nlvr2_test1.json')
    elif name == 'val+test1':
        if args.eval_data_type == 'bal':
            examples = processor.get_dev_examples(args.data_dir, args.use_label_seq, 'nlvr2_balanced_dev.json')
        elif args.eval_data_type == 'unbal':
            examples = processor.get_dev_examples(args.data_dir, args.use_label_seq, 'nlvr2_unbalanced_dev.json')
        else:
            examples = processor.get_dev_examples(args.data_dir, args.use_label_seq, 'nlvr2_dev_test1.json')

    return examples, labels


def _load_img_features(args):
    t_start = time.time()

    if args.img_feature_type == 'faster_r-cnn':  # faster_r-cnn
        if args.img_feature_dim == 2048:  # object features: 2048
            feat_file_name = 'nlvr2_img_frcnn_obj_feats.pt'
        else:  # object + spatial features: 2054
            feat_file_name = 'nlvr2_img_frcnn_feats.pt'  # nlvr2_img_frcnn_feats.pt
    else:
        feat_file_name = 'nlvr2_img_feats.pt'
    img_features = torch.load(os.path.join(args.data_dir, feat_file_name))

    t_end = time.time()
    logger.info('Info: loading {0:s} features using {1:.2f} secs'.format(feat_file_name, (t_end - t_start)))
    return img_features


def train(args, train_dataset, eval_dataset, student_model, teacher_model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  num_workers=args.workers,  # 0
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)  # original
    if args.optim == 'AdamW':
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optim == 'Adamax':
        optimizer = torch.optim.Adamax(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)  # original
    if args.scheduler == "constant":  # constant warmup and decay
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":  # linear warmup and decay
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # apex fp16 initialization
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        student_model, optimizer = amp.initialize(student_model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    student_model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    set_seed(args.seed, args.n_gpu)  # Added here for reproducibility (even between python 2 and 3)

    best_score = 0
    best_model = {
        'epoch': 0,
        'model': copy.deepcopy(student_model),  # student_model.state_dict()
        # 'optimizer': optimizer.state_dict()
    }

    # Prepare loss function
    mse_loss_fn = nn.MSELoss()

    def soft_cross_entropy(predictions, targets):
        student_likelihood = F.log_softmax(predictions, dim=-1)
        targets_probs = F.softmax(targets, dim=-1)
        return (-targets_probs * student_likelihood).mean()
    #

    # for epoch in range(int(args.num_train_epochs)):
    for epoch in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        t_start = time.time()

        # for step, batch in enumerate(train_dataloader):
        for step, batch in enumerate(epoch_iterator):
            # TinyBERT
            student_model.train()
            teacher_model.eval()

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                'labels': batch[3],
                'img_feats': None if args.img_feature_dim == -1 else batch[4]
            }

            task_specific_loss, student_logits, student_reps, student_atts = student_model(input_ids=inputs['input_ids'],
                                                                                           token_type_ids=inputs['token_type_ids'],
                                                                                           attention_mask=inputs['attention_mask'],
                                                                                           labels=inputs['labels'],
                                                                                           img_feats=inputs['img_feats'])
            with torch.no_grad():
                teacher_logits, teacher_reps, teacher_atts = teacher_model(input_ids=inputs['input_ids'],
                                                                           token_type_ids=inputs['token_type_ids'],
                                                                           attention_mask=inputs['attention_mask'],
                                                                           img_feats=inputs['img_feats'])

            teacher_layer_num = len(teacher_atts)
            student_layer_num = len(student_atts)
            assert teacher_layer_num % student_layer_num == 0
            layers_per_block = int(teacher_layer_num / student_layer_num)

            att_loss = 0.
            rep_loss = 0.

            # L_attn
            new_teacher_atts = [
                teacher_atts[i * layers_per_block + layers_per_block - 1] for i in range(student_layer_num)
            ]
            for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(args.device), student_att)
                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(args.device), teacher_att)
                tmp_loss = mse_loss_fn(student_att, teacher_att)
                att_loss += tmp_loss

            # L_embd + L_hidn
            new_teacher_reps = [
                teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)
            ]
            for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
                tmp_loss = mse_loss_fn(student_rep, teacher_rep)
                rep_loss += tmp_loss

            # L_pred = ?? * L_vanilla_kd + (1 - ??) * L_task_specific
            vanilla_kd_loss = soft_cross_entropy(student_logits / args.temperature, teacher_logits / args.temperature)
            cls_loss = (1 - args.alpha) * task_specific_loss + args.alpha * vanilla_kd_loss  # L(x;??_s;??_t)

            # L_s = ?? * (L_embd + L_hidn + L_attn) + L_pred
            loss = args.beta * (rep_loss + att_loss) + cls_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()  # ???_??_s L(x;??_s;??_t)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)  # 1.0

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        logger.info("***** Epoch: {} *****".format(epoch + 1))
        logger.info("  Train Loss: {}".format(tr_loss / len(train_dataset)))

        t_end = time.time()
        logger.info('  Train Time Cost: %.3f' % (t_end - t_start))

        # evaluation
        eval_result, eval_score = evaluate(args, student_model, eval_dataset, prefix='')
        if eval_score > best_score:
            best_score = eval_score
            best_model['epoch'] = epoch + 1
            best_model['model'] = copy.deepcopy(student_model)
            # best_model['optimizer'] = copy.deepcopy(optimizer.state_dict())
        # save checkpoints
        if (args.local_rank in [-1, 0]) and (args.save_epoch > 0 and epoch % args.save_epoch == 0) and (epoch > args.save_after_epoch):
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch + 1))  # ./model/nlvr2/student/checkpoint-1
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = student_model.module if hasattr(student_model, 'module') else student_model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            tokenizer.save_pretrained(output_dir)
            logger.info("Saving model checkpoint {0} to {1}".format(epoch + 1, output_dir))

        epoch_log = {'epoch': epoch + 1, 'eval_score': eval_score, 'best_score': best_score}
        log_json.append(epoch_log)

        if args.local_rank in [-1, 0]:
            with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                json.dump(log_json, fp)

        t_end = time.time()
        logger.info('Epoch: %d, Train Time: %.3f' % (epoch + 1, t_end - t_start))
        logger.info('********************')

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    # Save the final model checkpoint
    if args.local_rank in [-1, 0]:
        output_dir = args.output_dir  # model/nlvr2/student
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(student_model, 'module') else best_model['model']  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        tokenizer.save_pretrained(output_dir)
        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset=None, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     num_workers=args.workers,  # 0
                                     sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0

        correct_num = 0

        # for batch in eval_dataloader:
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()

            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],  # input_ids: (16, 2, 128)
                    'attention_mask': batch[1],  # input_mask: (16, 2, 178)
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # segment_ids: (16, 2, 128)
                    'labels': batch[3],  # label_id: (16, 1)
                    'img_feats': None if args.img_feature_dim == -1 else batch[4]  # img_feat: (16, 2, 50, 2054)
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

                correct = logits.argmax(1) == batch[3].view(-1)
                correct_num += correct.sum().item()

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        logger.info("  Eval Loss = %f" % eval_loss)

        acc = float(correct_num) / len(eval_dataloader.dataset)
        logger.info("  Eval Accuracy: {}".format(100 * acc))
        logger.info("  EVALERR: {:.2f}%".format(100 * acc))

        results.update({"acc": acc})

    t_end = time.time()
    logger.info('  Eval Time Cost: %.3f' % (t_end - t_start))

    return results, acc


def test(args, model, eval_dataset=None, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    label2ans = cPickle.load(open(args.label2ans_file, 'rb'))
    logger.info('label2ans: %d' % (len(label2ans)))

    results = []
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Test!
        logger.info("***** Running test {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        # for batch in eval_dataloader:
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()

            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                    'labels': None,
                    'img_feats': None if args.img_feature_dim == -1 else batch[4]
                }
                outputs = model(**inputs)

                logits = outputs[0]

                val, idx = logits.max(dim=1)

                for i in range(idx.size(0)):
                    result = {
                        'questionId': str(batch[5][i].item()),
                        'prediction': label2ans[eval_dataset.labels[idx[i].item()]]
                    }
                    results.append(result)

        with open(args.output_dir + ('/{}_results.json'.format(eval_dataset.name)), 'w') as fp:  # model/nlvr2/teacher/xxx_results.json
            json.dump(results, fp)

    t_end = time.time()
    logger.info('# questions: %d' % (len(results)))
    logger.info('Test Time Cost: %.3f' % (t_end - t_start))


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    # parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))

    # Text
    # parser.add_argument("--label_file", type=str, default=None, help="Label Dictionary")
    # parser.add_argument("--label2ans_file", type=str, default=None, help="Label to Answer Dictionary")
    # parser.add_argument("--data_label_type", default='faster', type=str, help="faster or mask")
    parser.add_argument("--eval_data_type", default='bal', type=str, help="bal or unbal or all")
    parser.add_argument("--test_data_type", default='bal', type=str, help="bal or unbal or all")

    # Image
    # parser.add_argument("--img_feat_dir", default=None, type=str, help="The input img_feat_dir.")
    # parser.add_argument("--img_feat_format", default='pt', type=str, help="img_feat_format: pt or tsv.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='faster_r-cnn', type=str, help="faster_r-cnn or mask_r-cnn")
    parser.add_argument("--max_img_seq_length", default=30, type=int, help="The maximum total input image sequence length.")
    parser.add_argument("--code_voc", default=512, type=int, help="dis_code_voc: 256, 512")
    parser.add_argument("--code_level", default='top', type=str, help="code level: top, bottom, both")

    # Dataset
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')

    # Model configuration
    parser.add_argument("--loss_type", default='ce', type=str, help="bce or ce")
    parser.add_argument("--classifier", default='linear', type=str, help="linear or mlp")
    parser.add_argument("--cls_hidden_scale", default=2, type=int, help="cls_hidden_scale: for classifier")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out for BERT.")
    # parser.add_argument("--use_img_layernorm", action='store_true', help="use_img_layernorm")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--optim", default='AdamW', type=str, help="optim: AdamW, Adamax")
    parser.add_argument("--use_pair", action='store_true', help="use_pair")
    parser.add_argument("--use_label_seq", action='store_true', help="use_label_seq")
    parser.add_argument("--num_choice", default=2, type=int, help="num_choice")

    # Other Parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run test on the test set.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Rul evaluation during training at each logging step.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=-1, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_epoch', type=int, default=1, help="Save checkpoint every X epochs.")
    parser.add_argument('--save_after_epoch', type=int, default=-1, help="Save checkpoint after epoch.")
    parser.add_argument("--eval_all_checkpoints", action='store_true', help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--philly", action='store_true', help="Use Philly: reset the output dir")

    #
    parser.add_argument("--teacher_model", default=None, type=str, help="The teacher model dir.")
    parser.add_argument("--student_model", default=None, type=str, required=True, help="The student model dir.")
    parser.add_argument('--alpha', default=0.5, type=float, help="Vanilla knowledge distillation loss radio.")
    parser.add_argument("--temperature", default=5.0, type=float, help="Distillation temperature for soft target.")
    parser.add_argument('--num_hidden_layers', default=6, type=int, help="Number of layers of the student model")
    parser.add_argument('--beta', default=0.01, type=float, help="intermediate features radio.")

    args = parser.parse_args()

    if args.philly:  # use philly
        logger.info('Info: Use Philly, all the output folders are reset.')
        args.output_dir = os.path.join(os.getenv('PT_OUTPUT_DIR'), args.output_dir)
        logger.info('OUTPUT_DIR:', args.output_dir)

    # Create or inspect output dir
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #     raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        logger.info("Output Directory Exists.")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s", args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed, args.n_gpu)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    logger.info('Task Name: {}, #Labels: {}'.format(args.task_name, num_labels))

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()  # bert
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]  # (BertConfig, ImageBertForSequenceClassification, BertTokenizer)
    if args.use_pair:  # True
        model_class = ImageBertForMultipleChoice  # (BertConfig, ImageBertForMultipleChoice, BertTokenizer)
    tokenizer = tokenizer_class.from_pretrained(  # BertTokenizer
        args.teacher_model,  # model/nlvr2/teacher
        do_lower_case=args.do_lower_case
    )
    teacher_config = config_class.from_pretrained(  # BertConfig
        args.teacher_model,  # model/nlvr2/teacher
        num_labels=num_labels,
        finetuning_task=args.task_name
    )
    student_config = config_class.from_pretrained(
        args.student_model,  # pretrained_models/base-vg-labels/ep_107_1192087
        num_hidden_layers=args.num_hidden_layers,
        num_labels=num_labels,
        finetuning_task=args.task_name
    )

    # new config: discrete code
    teacher_config.img_feature_dim = args.img_feature_dim
    teacher_config.img_feature_type = args.img_feature_type
    teacher_config.code_voc = args.code_voc
    teacher_config.hidden_dropout_prob = args.drop_out
    teacher_config.loss_type = args.loss_type
    teacher_config.classifier = args.classifier
    teacher_config.cls_hidden_scale = args.cls_hidden_scale
    # teacher_config.use_img_layernorm = args.use_img_layernorm
    teacher_config.num_choice = args.num_choice
    #
    teacher_config.output_hidden_states = True
    teacher_config.output_attentions = True

    student_config.img_feature_dim = args.img_feature_dim
    student_config.img_feature_type = args.img_feature_type
    student_config.code_voc = args.code_voc
    student_config.hidden_dropout_prob = args.drop_out
    student_config.loss_type = args.loss_type
    student_config.classifier = args.classifier
    student_config.cls_hidden_scale = args.cls_hidden_scale
    # student_config.use_img_layernorm = args.use_img_layernorm
    student_config.num_choice = args.num_choice
    #
    student_config.output_hidden_states = True
    student_config.output_attentions = True

    # # load discrete code
    # if args.img_feature_type in ['dis_code', 'dis_code_t']:
    #     logger.info('Load discrete code from: {}'.format(args.data_dir))
    #     t_start = time.time()
    #     train_code = torch.load(os.path.join(args.data_dir, 'vqvae', 'train.pt'))
    #     t_end = time.time()
    #     logger.info('Load time: %.3f' % (t_end - t_start))
    #
    #     if args.code_level == 'top':
    #         config.code_dim = train_code['embeddings_t'].shape[0]
    #         config.code_size = train_code['feats_top'][list(train_code['feats_top'].keys())[0]].shape[0]
    #     elif args.code_level == 'bottom':
    #         config.code_dim = train_code['embeddings_b'].shape[0]
    #         config.code_size = train_code['feats_bottom'][list(train_code['feats_bottom'].keys())[0]].shape[0]
    #     elif args.code_level == 'both':
    #         config.code_dim = train_code['embeddings_t'].shape[0] + train_code['embeddings_b'].shape[0]

    teacher_model = model_class.from_pretrained(  # ImageBertForMultipleChoice
        args.teacher_model,  # model/nlvr2/teacher
        from_tf=bool('.ckpt' in args.teacher_model),
        config=teacher_config
    )
    student_model = model_class.from_pretrained(  # ImageBertForMultipleChoice
        args.student_model,  # pretrained_models/base-vg-labels/ep_107_1192087
        from_tf=bool('.ckpt' in args.student_model),
        config=student_config
    )

    # if args.img_feature_type in ['dis_code', 'dis_code_t']:
    #     logger.info('Initializing the code embedding with {}'.format(args.code_level))
    #     if args.code_level == 'top':
    #         model.init_code_embedding(train_code['embeddings_t'].t())
    #     elif args.code_level == 'bottom':
    #         model.init_code_embedding(train_code['embeddings_b'].t())

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    teacher_total_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info('Teacher Model Parameters: {}'.format(teacher_total_params))
    student_total_params = sum(p.numel() for p in student_model.parameters())
    logger.info('Student Model Parameters: {}'.format(student_total_params))

    teacher_model.to(args.device)
    student_model.to(args.device)

    logger.info("Training/Evaluation parameters %s", args)

    # load image features
    img_features = _load_img_features(args)

    # Training (on 'train' set)
    if args.do_train:
        train_dataset = NLVRDataset(args, 'train', tokenizer, img_features)
        eval_dataset = NLVRDataset(args, 'val', tokenizer, img_features)
        global_step, tr_loss = train(args, train_dataset, eval_dataset, student_model, teacher_model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # # Evaluation (on 'val' set)
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     eval_dataset = NLVRDataset(args, 'val', tokenizer, img_features)
    #
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint, config=config)
    #         model.to(args.device)
    #         result, score = evaluate(args, model, eval_dataset, prefix=global_step)
    #
    # # Testing (on 'test1' set)
    # if args.do_test and args.local_rank in [-1, 0]:
    #     test_dataset = NLVRDataset(args, 'test1', tokenizer, img_features)
    #
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(args.device)
    #         # test(args, model, test_dataset, prefix=global_step)
    #         result, score = evaluate(args, model, test_dataset, prefix=global_step)


if __name__ == "__main__":
    main()
