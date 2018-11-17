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


def build_word_index(input_dir, output_json):
    word_count = OrderedDict()
    for root, dirs, files in walk(input_dir):
        for filename in files:
            if re.match(".*\d+_\d+.txt", filename):
                filepath = root + '/' + filename
                print(filepath)
                if 'unsup' in filepath:
                    continue
                with open(filepath, 'r') as f:
                    for line in f:
                        if six.PY2:
                            tokenize_words = tokenizer.tokenize(
                                    line.decode('utf-8').strip())
                        else:
                            tokenize_words = tokenizer.tokenize(line.strip())
                        lower_words = [word.lower() for word in tokenize_words]
                        for word in lower_words:
                            if word not in word_count:
                                word_count[word] = 0
                            word_count[word] += 1
    words = list(word_count.keys())
    counts = list(word_count.values())

    sorted_idx = np.argsort(counts)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    word_index = OrderedDict()
    for ii, ww in enumerate(sorted_words):
        word_index[ww] = ii + 1

    with open(output_json, 'w') as fp:
        json.dump(word_index, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--input_dir', type=str, nargs='?',
                        default='./data/aclImdb/',
                        help='input data directory')
    parser.add_argument('-ot', '--output_json', type=str, nargs='?',
                        default='./data/aclimdb_word_index.json',
                        help='output word index dict json')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_json = args.output_json
    build_word_index(input_dir, output_json)
