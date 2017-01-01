# coding: utf-8
'''
Preprocessing.
Note:
You need to create your own parallel Hangul-Hanja corpus first and
put it in `corpus/corpus.tsv`.
Each line must look like `Hangul sentence[Tab]Hanja sentence`.
For example,
나는 오늘 학교에 간다    나는 오늘 學校에 간다
This file should create the following files from the corpus.
`data/X.npy`: vectorized hangul sentences
`data/Y.npy`: vectorized hanja sentences
`data/charmaps.pkl`: 4 python dictionaries of character-index collections.
'''

import numpy as np
import cPickle as pickle
import codecs
import re

class Hyperparams:
    '''Hyper parameters'''
    batch_size = 64
    embed_dim = 200
    maxlen = 50
    hidden_dim = 200

def build_vocab():
    from collections import Counter
    from itertools import chain
    
    # romajis
    romaji_sents = [line.split('\t')[0] for line in codecs.open('E:/ja/ja.txt', 'r', 'utf-8').read().splitlines()]
    romajis = set(chain.from_iterable(romaji_sents))
    romajis = ["<EMP>", "<UNK>"] + list(romajis)
    romaji2idx = {romaji:idx for idx, romaji in enumerate(romajis)}
    idx2romaji = {idx:romaji for idx, romaji in enumerate(romajis)}
    
    # surfaces
    surface_sents = [line.split('\t')[1] for line in codecs.open('E:/ja/ja.txt', 'r', 'utf-8').read().splitlines()]
    surface2cnt = Counter(chain.from_iterable(surface_sents))
    surfaces = [surface for surface, cnt in surface2cnt.items() if cnt > 1] # remove singleton surfaces
    surfaces.remove("_")
    surfaces = ["<EMP>", "<UNK>", "_" ] + surfaces
    surface2idx = {surface:idx for idx, surface in enumerate(surfaces)}
    idx2surface = {idx:surface for idx, surface in enumerate(surfaces)}
    
    pickle.dump((romaji2idx, idx2romaji, surface2idx, idx2surface), open('E:/ja/vocab.pkl', 'wb'))

def load_vocab():
    romaji2idx, idx2romaji, surface2idx, idx2surface = pickle.load(open('E:/ja/vocab.pkl', 'rb'))
    return romaji2idx, idx2romaji, surface2idx, idx2surface
            
def prepro():
    '''Embeds and vectorize words in corpus'''
    build_vocab()
    romaji2idx, idx2romaji, surface2idx, idx2surface = load_vocab()
    
    print "romaji vocabulary size is", len(romaji2idx)
    print "surface vocabulary size is", len(surface2idx)
    
    print "# Vectorize"
    romaji_sents = [line.split('\t')[0] for line in codecs.open('E:/ja/ja.txt', 'r', 'utf-8').read().splitlines()]
    surface_sents = [line.split('\t')[1] for line in codecs.open('E:/ja/ja.txt', 'r', 'utf-8').read().splitlines()]
    
    xs, ys = [], [] # vectorized sentences
    for romaji_sent, surface_sent in zip(romaji_sents, surface_sents):
        if len(romaji_sent) <= Hyperparams.maxlen:
            x, y = [], []
            for romaji in romaji_sent:
                if romaji in romaji2idx:
                    x.append(romaji2idx[romaji])
                else:
                    x.append(1) #"OOV", i.e., not converted.
            
            for surface in surface_sent:
                if surface in surface2idx:
                    y.append(surface2idx[surface])
                else:
                    y.append(1) #"UNK", i.e., cannot converted. =>*
            
            x.extend([0] * (Hyperparams.maxlen - len(x))) # zero post-padding
            y.extend([0] * (Hyperparams.maxlen - len(y))) # zero post-padding
            
            xs.append(x) 
            ys.append(y) 
 
    print "# Convert to 2d-arrays"    
    X = np.array(xs)
    Y = np.array(ys)
    
    print "X.shape =", X.shape
    print "Y.shape =", Y.shape
    
    np.savez('E:/zh/X_Y.npz', X=X, Y=Y)
             
def load_data():
    '''Loads vectorized input training data
    '''
    X, Y = np.load('E:/ja/X_Y.npz')['X'], np.load('E:/ja/X_Y.npz')['Y']
    return X, Y

if __name__ == '__main__':
    prepro()
    print "Done" 