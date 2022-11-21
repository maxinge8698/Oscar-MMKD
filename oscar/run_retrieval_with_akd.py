# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

from __future__ import absolute_import, division, print_function

import argparse
import copy
import json
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

sys.path.insert(0, '.')

from transformers.pytorch_transformers import BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from transformers.pytorch_transformers import WEIGHTS_NAME

from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
from oscar.utils.misc import set_seed, mkdir
from oscar.utils.logger import setup_logger
from oscar.utils.task_utils import _truncate_seq_pair

import warnings

warnings.filterwarnings('ignore')


class RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""

    def __init__(self, args, tokenizer, split='train', is_train=True):
        super(RetrievalDataset, self).__init__()

        self.args = args
        self.tokenizer = tokenizer
        self.output_mode = args.output_mode
        self.is_train = is_train

        feature_file = os.path.join(args.data_dir, '{}_img_{}_feats.pt'.format(split, args.img_feature_type))
        self.features = torch.load(feature_file)

        caption_file = os.path.join(args.data_dir, '{}_captions.pt'.format(split))
        self.captions = torch.load(caption_file)

        self.img_keys = list(self.features.keys())

        if not type(self.captions[self.img_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}
        assert len(self.features) == len(self.captions), "the length of image features and captions does not match!"

        if args.add_od_labels:
            label_file = os.path.join(args.data_dir, '{}_{}_labels.pt'.format(split, args.od_label_type))
            self.labels = torch.load(label_file)

        if is_train:
            self.num_captions_per_img = args.num_captions_per_img_train
        else:
            self.num_captions_per_img = args.num_captions_per_img_val
            if args.eval_img_keys_file:
                # select a subset of image keys for evaluation. eg. COCO 1k and 5k
                # eval_img_keys_file is a list of image keys saved in tsv file
                with open(os.path.join(args.data_dir, args.eval_img_keys_file), 'r') as f:
                    img_keys = f.readlines()
                self.img_keys = [int(k.strip()) for k in img_keys]
                self.features = {k: self.features[k] for k in self.img_keys}
                self.captions = {k: self.captions[k] for k in self.img_keys}
                if args.add_od_labels:
                    self.labels = {k: self.labels[k] for k in self.img_keys}
            if args.eval_caption_index_file:
                # hard negative image/caption indexs for retrieval re-rank setting.
                # useful for mini val set to monitor the performance during training.
                # However, it cannot be used together with cross image evaluation.
                self.has_caption_indexs = True
                assert not args.cross_image_eval
                caption_index_file = os.path.join(args.data_dir, args.eval_caption_index_file)
                self.caption_indexs = torch.load(caption_index_file)
                if not type(self.caption_indexs[self.img_keys[0]]) == list:
                    self.caption_indexs = {k: json.loads(self.caption_indexs[k]) for k in self.img_keys}
            else:
                self.has_caption_indexs = False

    def get_image_caption_index(self, index):
        # return img_idx to access features and [img_key, cap_idx] to access caption
        if not self.is_train and self.args.cross_image_eval:
            img_idx = index // (self.num_captions_per_img * len(self.img_keys))
            cap_idx = index % (self.num_captions_per_img * len(self.img_keys))
            img_idx1 = cap_idx // self.num_captions_per_img
            cap_idx1 = cap_idx % self.num_captions_per_img
            return img_idx, [self.img_keys[img_idx1], cap_idx1]
        if not self.is_train and self.has_caption_indexs:
            img_idx = index // self.num_captions_per_img
            cap_idx = index % self.num_captions_per_img
            img_key1, cap_idx1 = self.caption_indexs[self.img_keys[img_idx]][cap_idx]
            return img_idx, [img_key1, cap_idx1]
        img_idx = index // self.num_captions_per_img
        cap_idx = index % self.num_captions_per_img
        return img_idx, [self.img_keys[img_idx], cap_idx]

    def get_label(self, index):
        img_idx, cap_idx = self.get_image_caption_index(index)
        return 1 if self.img_keys[img_idx] == cap_idx[0] else 0

    def get_od_labels(self, img_key):
        if self.args.add_od_labels:
            if type(self.labels[img_key]) == str:
                od_labels = self.labels[img_key]
            else:
                od_labels = ' '.join([l['class'] for l in self.labels[img_key]])
            return od_labels

    def tensorize_example(self,
                          text_a,
                          img_feat,
                          text_b=None,
                          cls_token_segment_id=0,
                          pad_token_segment_id=0,
                          sequence_a_segment_id=0,
                          sequence_b_segment_id=1):
        tokens_a = self.tokenizer.tokenize(text_a)

        if len(tokens_a) > self.args.max_seq_length - 2:
            tokens_a = tokens_a[:(self.args.max_seq_length - 2)]
        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        seq_a_len = len(tokens)

        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.args.max_seq_length - len(tokens) - 1:
                tokens_b = tokens_b[: (self.args.max_seq_length - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        seq_padding_len = self.args.max_seq_length - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.args.max_img_seq_length:
            img_feat = img_feat[0: self.args.max_img_seq_length, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.args.max_img_seq_length - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        att_mask_type = self.args.att_mask_type
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len + [0] * seq_padding_len + [1] * img_len + [0] * img_padding_len
        else:
            # use 2D mask to represent the attention
            max_len = self.args.max_seq_length + self.args.max_img_seq_length
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention of C-C, L-L, R-R
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            r_start, r_end = self.args.max_seq_length, self.args.max_seq_length + img_len
            attention_mask[c_start: c_end, c_start: c_end] = 1
            attention_mask[l_start: l_end, l_start: l_end] = 1
            attention_mask[r_start: r_end, r_start: r_end] = 1
            if att_mask_type == 'CL':
                attention_mask[c_start: c_end, l_start: l_end] = 1
                attention_mask[l_start: l_end, c_start: c_end] = 1
            elif att_mask_type == 'CR':
                attention_mask[c_start: c_end, r_start: r_end] = 1
                attention_mask[r_start: r_end, c_start: c_end] = 1
            elif att_mask_type == 'LR':
                attention_mask[l_start: l_end, r_start: r_end] = 1
                attention_mask[r_start: r_end, l_start: l_end] = 1
            else:
                raise ValueError("Unsupported attention mask type {}".format(att_mask_type))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return input_ids, attention_mask, segment_ids, img_feat

    def __getitem__(self, index):
        if self.is_train:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.img_keys[img_idx]
            feature = self.features[img_key]
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            od_labels = self.get_od_labels(img_key)
            example = self.tensorize_example(text_a=caption, img_feat=feature, text_b=od_labels)

            # select a negative pair
            neg_img_indexs = list(range(0, img_idx)) + list(range(img_idx + 1, len(self.img_keys)))
            img_idx_neg = random.choice(neg_img_indexs)
            if random.random() <= 0.5:
                # randomly select a negative caption from a different image.
                cap_idx_neg = random.randint(0, self.num_captions_per_img - 1)
                caption_neg = self.captions[self.img_keys[img_idx_neg]][cap_idx_neg]
                example_neg = self.tensorize_example(text_a=caption_neg, img_feat=feature, text_b=od_labels)
            else:
                # randomly select a negative image 
                feature_neg = self.features[self.img_keys[img_idx_neg]]
                od_labels_neg = self.get_od_labels(self.img_keys[img_idx_neg])
                example_neg = self.tensorize_example(text_a=caption, img_feat=feature_neg, text_b=od_labels_neg)

            example_pair = tuple(list(example) + [1] + list(example_neg) + [0])
            return index, example_pair
        else:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.img_keys[img_idx]
            feature = self.features[img_key]
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            od_labels = self.get_od_labels(img_key)
            example = self.tensorize_example(caption, feature, text_b=od_labels)

            label = 1 if img_key == cap_idxs[0] else 0
            return index, tuple(list(example) + [label])

    def __len__(self):
        if not self.is_train and self.args.cross_image_eval:
            return len(self.img_keys) ** 2 * self.num_captions_per_img
        return len(self.img_keys) * self.num_captions_per_img


def compute_score_with_logits(logits, labels):
    if logits.shape[1] > 1:
        logits = torch.max(logits, 1)[1].data
        scores = logits == labels
    else:
        scores = torch.zeros_like(labels).cuda()
        for i, (logit, label) in enumerate(zip(logits, labels)):
            logit_ = torch.sigmoid(logit)
            if (logit_ >= 0.5 and label == 1) or (logit_ < 0.5 and label == 0):
                scores[i] = 1
    return scores


def compute_ranks(dataset, results):
    labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
    similarities = np.array([results[i] for i in range(len(dataset))])
    if dataset.has_caption_indexs:
        num_captions_per_img = dataset.num_captions_per_img
    else:
        num_captions_per_img = len(dataset.img_keys) * dataset.num_captions_per_img
    labels = np.reshape(labels, [-1, num_captions_per_img])
    similarities = np.reshape(similarities, [-1, num_captions_per_img])
    i2t_ranks, t2i_ranks = [], []
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        i2t_ranks.append(rank)
    if not dataset.has_caption_indexs:
        labels = np.swapaxes(labels, 0, 1)
        similarities = np.swapaxes(similarities, 0, 1)
        for lab, sim in zip(labels, similarities):
            inds = np.argsort(sim)[::-1]
            rank = num_captions_per_img
            for r, ind in enumerate(inds):
                if lab[ind] == 1:
                    rank = r
                    break
            t2i_ranks.append(rank)
    return i2t_ranks, t2i_ranks


def save_checkpoint(model, tokenizer, args, epoch, global_step):
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(epoch, global_step))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while save_num < 10:
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, os.path.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return


def train(args, train_dataset, val_dataset, student_model, teacher_model, tokenizer, phase):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=args.num_workers)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        if phase == 1:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs_phase_1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs_phase_2

    # Prepare optimizer and scheduler
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
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    if args.n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    if phase == 1:
        logger.info("  Num Epochs = %d", args.num_train_epochs_phase_1)
    else:
        logger.info("  Num Epochs = %d", args.num_train_epochs_phase_2)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    train_acc = 0.0
    student_model.zero_grad()

    log_json = []

    best_score = 0
    best_model = {
        'epoch': 0,
        'model': copy.deepcopy(student_model),  # student_model.state_dict()
        # 'optimizer': optimizer.state_dict()
    }

    if phase == 1:
        num_train_epochs = args.num_train_epochs_phase_1
        temp = 1
    else:
        num_train_epochs = args.num_train_epochs_phase_2

    for epoch in range(int(num_train_epochs)):
        """
        If the phase is 1, then we need to increment the temperature after every (num_epochs_phase_1/max_temp) epochs
        For e.g. num_epochs_phase_1 = 20, max_temp = 10, then we increment temperature value after every 2 epochs
        """
        if phase == 1:
            if epoch % int(args.num_train_epochs_phase_1 / args.max_temperature) == 0 and epoch > 0:
                temp += 1
                logger.info(f"temperature is {temp}")

        for step, (_, batch) in enumerate(train_dataloader):
            # Annealing-KD
            student_model.train()
            teacher_model.eval()

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids': torch.cat((batch[0], batch[5]), dim=0),
                'attention_mask': torch.cat((batch[1], batch[6]), dim=0),
                'token_type_ids': torch.cat((batch[2], batch[7]), dim=0),
                'img_feats': torch.cat((batch[3], batch[8]), dim=0),
                'labels': torch.cat((batch[4], batch[9]), dim=0)
            }

            student_outputs = student_model(input_ids=inputs['input_ids'],
                                            token_type_ids=inputs['token_type_ids'],
                                            attention_mask=inputs['attention_mask'],
                                            # labels=inputs['labels'],
                                            img_feats=inputs['img_feats'])
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=inputs['input_ids'],
                                                token_type_ids=inputs['token_type_ids'],
                                                attention_mask=inputs['attention_mask'],
                                                img_feats=inputs['img_feats'])

            student_logits = student_outputs[0]
            teacher_logits = teacher_outputs[0]

            """
            Calculate loss based on phase:
                In Phase 1, use MSE loss between student and Annealed teacher output
                In Phase 2, use Cross-entropy loss between student and true labels
            """
            if phase == 1:
                loss = F.mse_loss(student_logits, teacher_logits * (temp / args.max_temperature))
            else:
                if args.loss_type == 'kl':
                    # KL Loss: https://github.com/uclanlp/visualbert/blob/master/pytorch_pretrained_bert/modeling.py
                    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
                    log_softmax = torch.nn.LogSoftmax(dim=-1)
                    reshaped_logits = student_logits.contiguous().view(-1, 3129)
                    reshaped_logits = log_softmax(reshaped_logits)
                    loss = loss_fct(reshaped_logits, inputs['labels'].contiguous())
                elif args.loss_type == 'bce':  # [VQA]
                    # loss = instance_bce_with_logits(logits, labels)
                    loss_fct = torch.nn.BCEWithLogitsLoss(reduction='mean')
                    loss = loss_fct(student_logits, inputs['labels'])
                    loss *= inputs['labels'].size(1)
                elif args.loss_type == 'ce':  # [GQA, Retrieval, Captioning]
                    loss_fct = nn.CrossEntropyLoss(reduction='mean')
                    loss = loss_fct(student_logits.view(-1, args.num_labels), input['labels'].view(-1))

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)  # 1.0

            batch_score = compute_score_with_logits(student_logits, inputs['labels']).sum()
            batch_acc = batch_score.item() / (args.train_batch_size * 2)
            train_acc += batch_acc

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % args.logging_steps == 0:
                logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), score: {:.4f} ({:.4f})".format(
                    epoch + 1, global_step, optimizer.param_groups[0]["lr"], loss, tr_loss / global_step, batch_acc, train_acc / global_step)
                )

            if (args.save_steps > 0 and global_step % args.save_steps == 0) or global_step == t_total:
                save_checkpoint(student_model, tokenizer, args, epoch + 1, global_step)
                # evaluation
                if args.evaluate_during_training:
                    logger.info("Perform evaluation at step: %d" % global_step)
                    test_result = test(args, student_model, val_dataset)
                    eval_result = evaluate(val_dataset, test_result)
                    rank_accs = eval_result['i2t_retrieval']
                    if rank_accs['R@1'] > best_score:
                        best_score = rank_accs['R@1']
                        best_model['epoch'] = epoch + 1
                        best_model['model'] = copy.deepcopy(student_model)
                    epoch_log = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'R1': rank_accs['R@1'],
                        'R5': rank_accs['R@5'],
                        'R10': rank_accs['R@10'],
                        'best_R1': best_score
                    }
                    log_json.append(epoch_log)
                    with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                        json.dump(log_json, fp)

    # Save the final model checkpoint
    if args.local_rank in [-1, 0]:
        output_dir = args.output_dir  # model/coco_ir/student
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(student_model, 'module') else best_model['model']  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        tokenizer.save_pretrained(output_dir)
        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))

    return global_step, tr_loss / global_step


def test(args, model, eval_dataset):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=args.num_workers)

    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))

    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    for indexs, batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'img_feats': batch[3],
                'labels': batch[4]
            }
            _, logits = model(**inputs)[:2]
            if args.num_labels == 2:
                probs = softmax(logits)
                result = probs[:, 1]  # the confidence to be a matched pair
            else:
                result = logits
            result = [_.to(torch.device("cpu")) for _ in result]
            results.update({idx.item(): res.item() for idx, res in zip(indexs, result)})
    return results


def evaluate(eval_dataset, test_results):
    i2t_ranks, t2i_ranks = compute_ranks(eval_dataset, test_results)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result


def get_predict_file(args):
    cc = []
    data = os.path.basename(os.path.join(args.data_dir, '')[:-1])
    if data != 'coco_ir':
        cc.append(data)
    cc.append(args.test_split)
    if args.add_od_labels:
        cc.append('wlabels{}'.format(args.od_label_type))
    return os.path.join(args.eval_model_dir, '{}.results.pt'.format('.'.join(cc)))


def restore_training_settings(args):
    assert not args.do_train and (args.do_test or args.do_eval)
    train_args = torch.load(os.path.join(args.eval_model_dir, 'training_args.bin'))
    override_params = ['do_lower_case', 'img_feature_type', 'max_seq_length', 'max_img_seq_length', 'add_od_labels', 'od_label_type']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param, test_v, train_v))
                setattr(args, param, train_v)
    return args


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default='datasets/coco_ir', type=str, required=False, help="The input data dir with all required files.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False, help="The output directory to save checkpoint and test results.")
    # parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    # parser.add_argument("--model_name_or_path", default='pretrained_models/base-vg-labels/ep_67_588997', type=str, required=False, help="Path to pre-trained model or model type. required for training.")

    # Text

    # Image
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str, help="Image feature type.")
    parser.add_argument("--max_img_seq_length", default=50, type=int, help="The maximum total input image sequence length.")

    # Dataset
    parser.add_argument("--num_workers", default=0, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_captions_per_img_train", default=5, type=int, help="number of positive matched captions for each training image.")
    parser.add_argument("--num_captions_per_img_val", default=5, type=int, help="number of captions for each testing image.")
    parser.add_argument("--att_mask_type", default='CLR', type=str, help="attention mask type, support ['CL', 'CR', 'LR', 'CLR'] C: caption, L: labels, R: image regions; CLR is full attention by default. CL means attention between caption and labels. please pay attention to the order CLR, which is the default concat order.")

    # Model configuration
    parser.add_argument("--loss_type", default='ce', type=str, help="Loss function types: support kl, ce")
    # parser.add_argument("--classifier", default='linear', type=str, help="linear or mlp")
    # parser.add_argument("--cls_hidden_scale", default=2, type=int, help="cls_hidden_scale: for classifier")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--output_mode", default='classification', type=str, help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int, help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--optim", default='AdamW', type=str, help="AdamW or Adamax")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=70, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. This number is calculated on COCO dataset, If add object detection labels, the suggested length should be 70.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run performance valuation. do not activate if we want to inference on dataset without gt labels.")
    # parser.add_argument("--num_train_epochs", default=20, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1, help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Run evaluation during training at each save_steps.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")

    # Training
    parser.add_argument("--eval_caption_index_file", default='', type=str, help="index of a list of (img_key, cap_idx) for each image. this is used to perform re-rank using hard negative samples. useful for validation set to monitor the performance during training.")
    parser.add_argument("--add_od_labels", default=False, action='store_true', help="Whether to add object detection labels or not.")
    parser.add_argument("--od_label_type", default='vg', type=str, help="label type, support vg, gt, oid")
    # Inference
    parser.add_argument("--test_split", default='test', type=str, help='data split name.')
    parser.add_argument("--eval_img_keys_file", default='', type=str, help="image key tsv to select a subset of images for evaluation. This is useful in 5-folds evaluation. The topn index file is not needed in this case.")
    parser.add_argument("--cross_image_eval", action='store_true', help="perform cross image inference, ie. each image with all texts from other images.")
    parser.add_argument("--eval_model_dir", type=str, default='', help="Model directory for evaluation.")

    #
    parser.add_argument("--teacher_model", default=None, type=str, help="The teacher model dir.")
    parser.add_argument("--student_model", default=None, type=str, required=True, help="The student model dir.")
    # parser.add_argument('--alpha', default=0.5, type=float, help="Vanilla knowledge distillation loss radio.")
    # parser.add_argument("--temperature", default=5.0, type=float, help="Distillation temperature for soft target.")
    parser.add_argument('--num_hidden_layers', default=6, type=int, help="Number of layers of the student model")
    parser.add_argument("--max_temperature", default=7, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_epochs_phase_1", default=14, type=int, help="Total number of training epochs in phase 1.")
    parser.add_argument("--num_train_epochs_phase_2", default=4, type=int, help="Total number of training epochs in phase 2.")

    args = parser.parse_args()

    global logger
    mkdir(args.output_dir)
    logger = setup_logger("vlpretrain", args.output_dir, 0)

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    logger.info('output_mode: {}, #Labels: {}'.format(args.output_mode, args.num_labels))

    config_class, model_class, tokenizer_class = BertConfig, ImageBertForSequenceClassification, BertTokenizer
    if args.do_train:
        tokenizer = tokenizer_class.from_pretrained(  # BertTokenizer
            args.teacher_model,  # model/coco_ir/teacher
            do_lower_case=args.do_lower_case
        )
        teacher_config = config_class.from_pretrained(  # BertConfig
            args.teacher_model,  # model/coco_ir/teacher
            num_labels=args.num_labels,  # 2
            finetuning_task='ir'
        )
        student_config = config_class.from_pretrained(  # BertConfig
            args.student_model,  # pretrained_models/base-vg-labels/ep_67_588997
            num_hidden_layers=args.num_hidden_layers,
            num_labels=args.num_labels,  # 2
            finetuning_task='ir'
        )

        teacher_config.img_feature_dim = args.img_feature_dim
        teacher_config.img_feature_type = args.img_feature_type
        teacher_config.hidden_dropout_prob = args.drop_out
        teacher_config.loss_type = args.loss_type

        student_config.img_feature_dim = args.img_feature_dim
        student_config.img_feature_type = args.img_feature_type
        student_config.hidden_dropout_prob = args.drop_out
        student_config.loss_type = args.loss_type

        teacher_model = model_class.from_pretrained(  # ImageBertForSequenceClassification
            args.teacher_model,  # model/coco_ir/teacher
            from_tf=bool('.ckpt' in args.teacher_model),
            config=teacher_config
        )
        student_model = model_class.from_pretrained(  # ImageBertForSequenceClassification
            args.student_model,  # pretrained_models/base-vg-labels/ep_67_588997
            from_tf=bool('.ckpt' in args.student_model),
            config=student_config
        )
    else:
        checkpoint = args.eval_model_dir
        assert os.path.isdir(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        config = config_class.from_pretrained(checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)

    teacher_total_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info('Teacher Model Parameters: {}'.format(teacher_total_params))
    student_total_params = sum(p.numel() for p in student_model.parameters())
    logger.info('Student Model Parameters: {}'.format(student_total_params))

    teacher_model.to(args.device)
    student_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = RetrievalDataset(args, tokenizer, 'train', is_train=True)
        if args.evaluate_during_training:
            eval_dataset = RetrievalDataset(args, tokenizer, 'minival', is_train=False)
        else:
            eval_dataset = None

        #
        # Train student model in phase 1
        print("*" * 100)
        print("*" + " " * 39 + "Phase 1 of training" + " " * 40 + "*")
        print("*" * 100)
        original_output_dir = args.output_dir
        args.output_dir = os.path.join(original_output_dir, 'phase1')
        global_step, tr_loss = train(args, train_dataset, eval_dataset, student_model, teacher_model, tokenizer, phase=1)

        # Train student model in phase 2
        print("*" * 100)
        print("*" + " " * 39 + "Phase 2 of training" + " " * 38 + "*")
        print("*" * 100)
        student_model = student_model.from_pretrained(args.output_dir)
        student_model.to(args.device)
        args.output_dir = original_output_dir
        global_step, tr_loss = train(args, train_dataset, eval_dataset, student_model, teacher_model, tokenizer, phase=2)
        #

        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    # # Inference and evaluation
    # if args.do_test or args.do_eval:
    #     args = restore_training_settings(args)
    #     test_dataset = RetrievalDataset(args, tokenizer, args.test_split, is_train=False)
    #     checkpoint = args.eval_model_dir
    #     assert os.path.isdir(checkpoint)
    #     logger.info("Evaluate the following checkpoint: %s", checkpoint)
    #
    #     model = model_class.from_pretrained(checkpoint, config=config)
    #     model.to(args.device)
    #     if args.n_gpu > 1:
    #         model = torch.nn.DataParallel(model)
    #
    #     pred_file = get_predict_file(args)
    #     if os.path.isfile(pred_file):
    #         logger.info("Prediction file exist, skip inference.")
    #         if args.do_eval:
    #             test_result = torch.load(pred_file)
    #     else:
    #         test_result = test(args, model, test_dataset)
    #         torch.save(test_result, pred_file)
    #         logger.info("Prediction results saved to {}.".format(pred_file))
    #
    #     if args.do_eval:
    #         eval_result = evaluate(test_dataset, test_result)
    #         result_file = os.path.splitext(pred_file)[0] + '.eval.json'
    #         with open(result_file, 'w') as f:
    #             json.dump(eval_result, f)
    #         logger.info("Evaluation results saved to {}.".format(result_file))


if __name__ == "__main__":
    main()
