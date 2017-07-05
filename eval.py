#-*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_vocab
from train import Graph
from utils import *
import distance

def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    nums, X, gts = load_test_data()
    roma2idx, idx2roma, surf2idx, idx2surf = load_vocab()
             
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
             
            # Get model
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
               
            with codecs.open('results/{}_{}_beam_width_{}.csv'.format(hp.norm_type, mname, hp.beam_width), 'w', 'utf-8') as fout:
                fout.write("NUM,EXPECTED,{}_beam_width_{},# characters,edit distance\n".format(mname, hp.beam_width))
                
                total_edit_distance = 0
                num_chars = 0 # number of total characters
                for step in range(len(X)//hp.batch_size):
                    num = nums[step*hp.batch_size:(step+1)*hp.batch_size] #number batch
                    x = X[step*hp.batch_size:(step+1)*hp.batch_size] # input batch
                    gt = gts[step*hp.batch_size:(step+1)*hp.batch_size] # batch of ground truth strings
                    
                    if hp.beam_width==1:
                        preds = np.zeros((hp.batch_size, hp.max_len), np.int32)
                        for j in range(hp.max_len):
                            _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                            preds[:, j] = _preds[:, j]
                    else: # beam decode    
                        ## first step  
                        preds = np.zeros((hp.beam_width*hp.batch_size, hp.max_len), np.int32)  #  (bw*N, T)
                        logprobs = sess.run(g.logprobs, {g.x: x, g.y: np.zeros((hp.batch_size, hp.max_len), np.int32)}) # (N, T, V)
                        target = logprobs[:, 0, :] # (N, V)
                        
                        preds_in_beam = target.argsort()[:, ::-1][:, :hp.beam_width].flatten() # (bw*N,)
                        preds[:, 0] = preds_in_beam
                        
                        logp_in_beam = np.sort(target)[:, ::-1][:, :hp.beam_width].flatten() # (bw*N,)
                        logp_in_beam = np.repeat(logp_in_beam, hp.beam_width, axis=0) # (bw*bw*N, )
                         
                        ## remaining steps
                        for i in range(1, hp.max_len-1):
                            logprobs = sess.run(g.logprobs, {g.x: np.repeat(x, hp.beam_width, 0), g.y: preds}) # (bw*N, T, V)
                            target = logprobs[:, i, :] # (bw*N, V)
                             
                            preds_in_beam = target.argsort()[:, ::-1][:, :hp.beam_width].flatten() # (bw*bw*N,)
                            logp_in_beam += np.sort(target)[:, ::-1][:, :hp.beam_width].flatten() # (bw*bw*N, )
     
                            preds = np.repeat(preds, hp.beam_width, axis=0) # (bw*bw*N, T) <- Temporary shape expansion
                            preds[:, i] = preds_in_beam
                                   
                            elems = [] # (bw*N). bw elements are selected out of bw^2
                            for j, cluster in enumerate(np.split(logp_in_beam, hp.batch_size)): # cluster: (bw*bw,)
                                if i == hp.max_len-2: # final step
                                    elem = np.argsort(cluster)[::-1][:1] # final 1 best
                                    elems.extend(list(elem + j*len(cluster)))
                                else:
                                    elem = np.argsort(cluster)[::-1][:hp.beam_width]
                                    elems.extend(list(elem + j*len(cluster)))
                            preds = preds[elems] # (N, T) if final step,  (bw*N, T) otherwise. <- shape restored
                            logp_in_beam = logp_in_beam[elems]
                            logp_in_beam = np.repeat(logp_in_beam, hp.beam_width, axis=0) # (bw*bw*N, )
                            
#                             for l, pred in enumerate(preds[:hp.beam_width]):
#                                 fout.write(str(l) + " " + u"".join(idx2surf[idx] for idx in pred).split("S")[0] + "\n")
                        
                    for n, pred, expected in zip(num, preds, gt): # sentence-wise
                        got = "".join(idx2surf[idx] for idx in pred).split("S")[0]
                         
                        edit_distance = distance.levenshtein(expected, got)
                        total_edit_distance += edit_distance
                        num_chars += len(expected)
                          
                        fout.write(u"{},{},{},{},{}\n".format(n, expected, got, len(expected), edit_distance))
                fout.write(u"Total CER: {}/{}={},,,,\n".format(total_edit_distance, 
                                                        num_chars, 
                                                        round(float(total_edit_distance)/num_chars, 2)))
if __name__ == '__main__':
    eval()
    print("Done")
    
    
