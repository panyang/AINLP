#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: TextMiner (textminer@foxmail.com)
# Copyright 2018 @ AINLP

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np
import re
import six

from collections import OrderedDict
from os import walk
from sacremoses import MosesTokenizer

tokenizer = MosesTokenizer()


def get_word_index(word_index_path):
    with open(word_index_path) as f:
        return json.load(f)


def build_data_index(input_dir, word_index):
    train_x = []
    train_y = []
    for root, dirs, files in walk(input_dir):
        for filename in files:
            if re.match(".*\d+_\d+.txt", filename):
                filepath = root + '/' + filename
                print(filepath)
                if 'pos' in filepath:
                    train_y.append(1)
                elif 'neg' in filepath:
                    train_y.append(0)
                else:
                    continue
                train_list = []
                with open(filepath, 'r') as f:
                    for line in f:
                        if six.PY2:
                            tokenize_words = tokenizer.tokenize(
                                    line.decode('utf-8').strip())
                        else:
                            tokenize_words = tokenizer.tokenize(line.strip())
                        lower_words = [word.lower() for word in tokenize_words]
                        for word in lower_words:
                            train_list.append(word_index.get(word, 0))
                train_x.append(train_list)
    return train_x, train_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-trd', '--train_dir', type=str, nargs='?',
                        default='./data/aclImdb/train/',
                        help='train data directory')
    parser.add_argument('-ted', '--test_dir', type=str, nargs='?',
                        default='./data/aclImdb/test/',
                        help='test data directory')
    parser.add_argument('-wip', '--word_index_path', type=str, nargs='?',
                        default='./data/aclimdb_word_index.json',
                        help='aclimdb word index json')
    parser.add_argument('-onz', '--output_npz', type=str, nargs='?',
                        default='./data/aclimdb.npz',
                        help='output npz')
    args = parser.parse_args()
    train_dir = args.train_dir
    test_dir = args.test_dir
    word_index_path = args.word_index_path
    output_npz = args.output_npz
    word_index = get_word_index(word_index_path)
    train_x, train_y = build_data_index(train_dir, word_index)
    test_x, test_y = build_data_index(test_dir, word_index)
    np.savez(output_npz,
             x_train=np.asarray(train_x),
             y_train=np.asarray(train_y),
             x_test=np.asarray(test_x),
             y_test=np.asarray(test_y))
