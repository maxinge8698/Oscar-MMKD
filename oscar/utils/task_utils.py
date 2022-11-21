# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

from __future__ import absolute_import, division, print_function

import _pickle as cPickle
import csv
import json
import logging
import os
import sys
from io import open

import torch

logger = logging.getLogger(__name__)


class InputInstance(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, score=None, img_key=None, q_id=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.score = score
        self.img_key = img_key
        self.q_id = q_id


class InputFeat(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, score, img_feat):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.score = score
        self.img_feat = img_feat


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class VQATextProcessor(DataProcessor):
    """ Processor for the VQA Text data set. """

    def get_train_examples(self, data_dir, file_name='train2014_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir, file_name='val2014_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "dev")

    def get_test_examples(self, data_dir, file_name='test2015_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "test")

    def get_labels(self, label_file):
        """ See base class."""

        ans2label = cPickle.load(open(label_file, 'rb'))
        label_list = list(ans2label.values())
        return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []
        for (i, line) in enumerate(lines):
            if set_type != 'test' and len(line['an']) == 0:
                continue

            guid = "%s-%s" % (set_type, str(i))
            text_a = line['q']
            text_b = line['o'].replace(';', ' ').strip()
            label = None if set_type.startswith('test') else line['an']
            score = None if set_type.startswith('test') else line['s']
            img_key = line['img_id']
            q_id = int(line['q_id']) if set_type.startswith('test') else 0
            examples.append(
                InputInstance(guid=guid, text_a=text_a, text_b=text_b, label=label, score=score, img_key=img_key, q_id=q_id)
            )
        return examples


class VQATextAProcessor(DataProcessor):
    """ Processor for the VQA Text data set. """

    def get_train_examples(self, data_dir, file_name='train2014_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir, file_name='val2014_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "dev")

    def get_test_examples(self, data_dir, file_name='test2015_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "test")

    def get_labels(self, label_file):
        """ See base class."""

        ans2label = cPickle.load(open(label_file, 'rb'))
        return list(ans2label.values())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []
        for (i, line) in enumerate(lines):
            if set_type != 'test' and len(line['an']) == 0: continue

            guid = "%s-%s" % (set_type, str(i))
            text_a = line['q']
            text_b = None  # line['o'] # or None
            label = None if set_type.startswith('test') else line['an']
            score = None if set_type.startswith('test') else line['s']
            img_key = line['img_id']
            q_id = int(line['q_id']) if set_type.startswith('test') else 0
            examples.append(InputInstance(guid=guid, text_a=text_a, text_b=text_b, label=label, score=score, img_key=img_key, q_id=q_id))
        return examples


class GQAProcessor(DataProcessor):
    """ Processor for the GQA data set. """

    def get_train_examples(self, data_dir, file_name='train2014_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir, file_name='val2014_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "dev")

    def get_test_examples(self, data_dir, file_name='test2015_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "test")

    def get_labels(self, label_file='trainval_testdev_all_ans2label.pkl'):
        """ See base class."""

        ans2label = cPickle.load(open(label_file, 'rb'))
        label_list = list(ans2label.values())
        return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []
        for (i, line) in enumerate(lines):
            if set_type != 'test' and len(line['an']) == 0:
                continue

            guid = "%s-%s" % (set_type, str(i))
            text_a = line['q']
            # text_b = line['o'] # or None
            text_b = line['o'].replace(';', ' ').strip()
            label = None if set_type.startswith('test') else line['an']
            score = None
            img_key = line['img_id']
            q_id = int(line['q_id']) if set_type.startswith('test') else 0
            examples.append(
                InputInstance(guid=guid, text_a=text_a, text_b=text_b, label=label, score=score, img_key=img_key, q_id=q_id)
            )
        return examples


class NLVRProcessor(DataProcessor):
    """ Processor for the NLVR data set. """

    def get_train_examples(self, data_dir, use_label_seq=True, file_name='nlvr2_train.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "train", use_label_seq)

    def get_dev_examples(self, data_dir, use_label_seq=True, file_name='nlvr2_dev.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "dev", use_label_seq)

    def get_test_examples(self, data_dir, use_label_seq=True, file_name='nlvr2_test1.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "test", use_label_seq)

    def get_labels(self, label_file=None):
        """ See base class."""

        # ans2label = cPickle.load(open(label_file, 'rb'))
        # label_list = list(ans2label.values())
        # return label_list
        return [0, 1]

    def _create_examples(self, lines, set_type, use_label_seq=True):
        """ Creates examples for the training and dev sets. """

        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, str(i))
            text_a = line['q']
            text_b = line['o'] if use_label_seq else None
            # label = None if set_type.startswith('test') else line['label']
            label = line['label']
            score = None
            img_key = line['img_id']
            # q_id = int(line['q_id']) if set_type.startswith('test') else 0
            q_id = 0
            examples.append(
                InputInstance(guid=guid, text_a=text_a, text_b=text_b, label=label, score=score, img_key=img_key, q_id=q_id)
            )
        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


processors = {
    "vqa_text": VQATextProcessor,
    "vqa_text_a": VQATextAProcessor,
    "gqa": GQAProcessor,
    "nlvr": NLVRProcessor
}

output_modes = {
    "vqa_text": "classification",
    "vqa_text_a": "classification",
    "gqa": "classification",
    "nlvr": "classification"
}

GLUE_TASKS_NUM_LABELS = {
    "vqa_text": 3129,
    "vqa_text_a": 3129,
    "gqa": 1853,
    "nlvr": 2
}
