# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
www.github.com/kyubyong/neural_japanese_transliterator
'''

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from utils import *
import codecs
import pickle

def load_vocab():
    def _make_dicts(fpath, min_cnt):
        tokens = []
        for line in codecs.open(fpath, 'r', 'utf-8'):
            token, cnt = line.strip("\n").split("\t")
            if len(cnt)==0 or (len(cnt) > 0 and int(cnt) >= min_cnt):
                tokens.append(token)
        token2idx = {token:idx for idx, token in enumerate(tokens)}
        idx2token = {idx:token for idx, token in enumerate(tokens)}
        return token2idx, idx2token
    
    # romaji vocab
    roma2idx, idx2roma = _make_dicts('preprocessed/vocab.romaji.txt', 5)
    surf2idx, idx2surf = _make_dicts('preprocessed/vocab.surface.txt', 5)
    
    return roma2idx, idx2roma, surf2idx, idx2surf

def load_train_data():
    '''Loads vectorized input training data
    '''
    romaji_sents, surface_sents = pickle.load(open('preprocessed/train.pkl', 'rb'))
    return romaji_sents, surface_sents

def load_test_data():
    '''Embeds and vectorize words in input corpus'''
    lines = [line for line in codecs.open('data/test.csv', 'r', 'utf-8').read().splitlines()[1:]]
     
    roma2idx, _, surf2idx, _ = load_vocab()
    
    nums = [] 
    xs = []
    gt = []
    for line in lines:
        num, romaji_sent, expected = line.split(",")
        
        nums.append(num)
        gt.append(expected)
        
        x = [roma2idx.get(romaji, 1) for romaji in romaji_sent + "S"]
        x += [0]*(hp.max_len-len(x))
        xs.append(x)
     
    X = np.array(xs)
    return nums, X, gt

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        romaji_sents, surface_sents = load_train_data() # byte
        
        # calc total batch count
        num_batch = len(romaji_sents) // hp.batch_size
         
        # Convert to tensor
        romaji_sents = tf.convert_to_tensor(romaji_sents)
        surface_sents = tf.convert_to_tensor(surface_sents)
         
        # Create Queues
        romaji_sents, surface_sents = tf.train.slice_input_producer([romaji_sents, surface_sents], shuffle=True)

        @producer_func
        def _restore(_inputs):
            '''Fetch the value from slice queues,
               then enqueue them again. 
            '''
            _romaji_sents, _surface_sents = _inputs
            
            # Processing
            _romaji_sents = np.fromstring(_romaji_sents, np.int32) # byte to int
            _surface_sents = np.fromstring(_surface_sents, np.int32) # byte to int
    
            return _romaji_sents, _surface_sents
            
        # Decode sound file
        x, y = _restore(inputs=[romaji_sents, surface_sents], 
                        dtypes=[tf.int32, tf.int32],
                        capacity=128,
                        num_threads=32)
        # create batch queues
        x, y = tf.train.batch([x, y],
                                shapes=[(None,), (None,)],
                                num_threads=32,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*32,   
                                dynamic_pad=True)
    return x, y, num_batch