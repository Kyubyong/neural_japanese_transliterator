# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
www.github.com/kyubyong/neural_japanese_transliterator
'''

class Hyperparams:
    '''Hyper parameters'''
    # data
    max_len = 50 # maximum length of text
    
    # model
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    norm_type = "ln"
    dropout_rate = 0.0
    
    # training scheme
    lr = 0.0001
    logdir = "logdir"
    batch_size = 32
    num_epochs = 5
    
    # inference
    beam_width = 1 # if beam width is 1, we apply a regular greedy decoding.
    
