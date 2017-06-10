# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
www.github.com/kyubyong/neural_japanese_transliterator
'''
from __future__ import print_function

import codecs
from collections import Counter
from itertools import chain
import os
import pickle
import re

from hyperparams import Hyperparams as hp
import numpy as np


def build_vocab():
    # Make romaji and surface (hiragana/katakana/kanji) sentences that are valid.
    romaji_sents, surface_sents = [], []
    for line in codecs.open('preprocessed/ja.tsv', 'r', 'utf-8'):
        try:
            idx, romaji_sent, surface_sent = line.strip().split("\t")
        except ValueError:
            continue
            
        if len(romaji_sent) < hp.max_len:
            romaji_sents.append(romaji_sent)
            surface_sents.append(surface_sent)
    
    # Make Romaji vocabulary
    with codecs.open('preprocessed/vocab.romaji.txt', 'w', 'utf-8') as fout:
        fout.write("E\t\nU\t\nS\t\n") #E: Empty, U: Unkown
        roma2cnt = Counter(chain.from_iterable(romaji_sents))
        for roma, cnt in roma2cnt.most_common(len(roma2cnt)):
            fout.write(u"{}\t{}\n".format(roma, cnt))
    
    # Make surface vocabulary
    with codecs.open('preprocessed/vocab.surface.txt', 'w', 'utf-8') as fout:
        fout.write("E\t\nU\t\nS\t\n") #E: Empty, U: Unkown
        surf2cnt = Counter(chain.from_iterable(surface_sents))
        for surf, cnt in surf2cnt.most_common(len(surface_sents)):
            fout.write(u"{}\t{}\n".format(surf, cnt))

def create_train_data():
    from data_load import load_vocab
    roma2idx, idx2roma, surf2idx, idx2surf = load_vocab()
    romaji_sents, surface_sents = [], []
    for line in codecs.open('preprocessed/ja.tsv', 'r', 'utf-8'):
        try:
            idx, romaji_sent, surface_sent = line.strip().split("\t")
        except ValueError:
            continue
        
        if len(romaji_sent) < hp.max_len:
            romaji_sents.append(np.array([roma2idx.get(roma, 1) for roma in romaji_sent+"S"], np.int32).tostring())
            surface_sents.append(np.array([surf2idx.get(surf, 1) for surf in surface_sent+"S"], np.int32).tostring())
    pickle.dump((romaji_sents, surface_sents), open('preprocessed/train.pkl', 'wb'), protocol=2)               
    
if __name__ == '__main__':
#     build_vocab()
    create_train_data()
    print("Done") 
