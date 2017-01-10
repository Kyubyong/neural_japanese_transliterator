# coding: utf-8
'''
Preprocessing.
'''
from __future__ import print_function 
import numpy as np
import cPickle as pickle
import codecs
import re

class Hyperparams:
    '''Hyper parameters'''
    batch_size = 64
    embed_dim = 300
    seqlen = 50

def build_vocab():
    from collections import Counter
    from itertools import chain
    
    # romajis
    romaji_sents = [line.split('\t')[1] for line in codecs.open('data/ja.tsv', 'r', 'utf-8').read().splitlines()]
    romajis = set(chain.from_iterable(romaji_sents))
    romajis = ["E", "U", "S"] + list(romajis) # E: Empty, U: Unknown, S: end of Sentence
    roma2idx = {romaji:idx for idx, romaji in enumerate(romajis)}
    idx2roma = {idx:romaji for idx, romaji in enumerate(romajis)}
    
    # surfaces
    surface_sents = [line.split('\t')[2] for line in codecs.open('data/ja.tsv', 'r', 'utf-8').read().splitlines()]
    surface2cnt = Counter(chain.from_iterable(surface_sents))
    surfaces = [surface for surface, cnt in surface2cnt.items() if cnt > 5] # remove infrequent surfaces
    surfaces = ["E", "U", "S"] + surfaces
    surf2idx = {surface:idx for idx, surface in enumerate(surfaces)}
    idx2surf = {idx:surface for idx, surface in enumerate(surfaces)}
    
    pickle.dump((roma2idx, idx2roma, surf2idx, idx2surf), open('data/vocab.pkl', 'wb'))

def load_vocab():
    roma2idx, idx2roma, surf2idx, idx2surf = pickle.load(open('data/vocab.pkl', 'rb'))
    return roma2idx, idx2roma, surf2idx, idx2surf
            
def create_train_data():
    '''Embeds and vectorize words in corpus'''
    roma2idx, idx2roma, surf2idx, idx2surf = load_vocab()
    
    print("# Vectorize")
    romaji_sents = [line.split('\t')[1] for line in codecs.open('data/ja.tsv', 'r', 'utf-8').read().splitlines()]
    surface_sents = [line.split('\t')[2] for line in codecs.open('data/ja.tsv', 'r', 'utf-8').read().splitlines()]
    
    xs, ys = [], [] # vectorized sentences
    for romaji_sent, surface_sent in zip(romaji_sents, surface_sents):
        if ( 10 < len(romaji_sent) < Hyperparams.seqlen ) and ( 10 < len(surface_sent) < Hyperparams.seqlen ):
            x, y = [], []
            for romaji in romaji_sent + "S":
                if romaji in roma2idx:
                    x.append(roma2idx[romaji])
                else:
                    x.append(1) #"UNK"
            
            for surface in surface_sent + "S":
                if surface in surf2idx:
                    y.append(surf2idx[surface])
                else:
                    y.append(1) #"UNK"
            
            x.extend([0] * (Hyperparams.seqlen - len(x))) # zero post-padding
            y.extend([0] * (Hyperparams.seqlen - len(y))) # zero post-padding
            
            xs.append(x) 
            ys.append(y) 
 
    print("# Convert to 2d-arrays")    
    X = np.array(xs)
    Y = np.array(ys)
    
    print("X.shape =", X.shape) # (114176, 50)
    print("Y.shape =", Y.shape) # (114176, 50)
    
    np.savez('data/X_Y.npz', X=X, Y=Y)
             
def load_train_data():
    '''Loads vectorized input training data
    '''
    X, Y = np.load('data/X_Y.npz')['X'], np.load('data/X_Y.npz')['Y']
    return X, Y

def load_test_data():
    '''Embeds and vectorize words in input corpus'''
    try:
        lines = [line for line in codecs.open('data/input.csv', 'r', 'utf-8').read().splitlines()[1:]]
    except IOError:
        raise IOError("Write the sentences you want to test line by line in `data/input.csv` file.")
     
    roma2idx, _, surf2idx, _ = load_vocab()
    
    nums = [] 
    xs = []
    expected_list = []
    for line in lines:
        num, romaji_sent, expected = line.split(",")
        
        nums.append(num)
        expected_list.append(expected)
        
        x = []
        for romaji in romaji_sent[:Hyperparams.seqlen - 1] + "S":
            if romaji in roma2idx: 
                x.append(roma2idx[romaji])
            else:
                x.append(1) #"OOV", i.e., not converted.
         
        x.extend([0] * (Hyperparams.seqlen - len(x))) # zero post-padding
        xs.append(x)
     
    X = np.array(xs)
    return nums, X, expected_list

if __name__ == '__main__':
    build_vocab()
    create_train_data()
    print("Done") 