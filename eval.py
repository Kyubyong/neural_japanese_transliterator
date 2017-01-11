#-*- coding: utf-8 -*-
#!/usr/bin/python2
'''
Test.
Given test sentences in `data/input.csv`,
it writes the results to `data/output_***.txt` 
'''
from __future__ import print_function
import sugartensor as tf
import numpy as np
from prepro import Hyperparams, load_vocab, load_test_data
from train import ModelGraph
import codecs
import pickle
import distance

def main():  
    g = ModelGraph(mode="test")
        
    with tf.Session() as sess:
        tf.sg_init(sess)

        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint("asset/train/ckpt"))
        mname = open("asset/train/ckpt/checkpoint", 'r').read().split('"')[1]
        
        nums, X, expected_list = load_test_data()
        roma2idx, idx2roma, surf2idx, idx2surf = load_vocab()
        
        with codecs.open('data/output_{}.txt'.format(mname), 'w', 'utf-8') as fout:
            cum_score = 0
            full_score = 0
            for step in range(len(X)//Hyperparams.batch_size):
                n = nums[step*Hyperparams.batch_size:(step+1)*Hyperparams.batch_size] #number batch
                x = X[step*Hyperparams.batch_size:(step+1)*Hyperparams.batch_size] # input batch
                e = expected_list[step*Hyperparams.batch_size:(step+1)*Hyperparams.batch_size] # batch of ground truth strings
                
                preds_prev = np.zeros((Hyperparams.batch_size, Hyperparams.seqlen), np.int64)
                preds = np.zeros((Hyperparams.batch_size, Hyperparams.seqlen), np.int64)
                
                i = 0
                while i < Hyperparams.seqlen - 1:
                    logits = sess.run(g.logits, {g.x: x, g.y_src: preds_prev})
                    pred = np.argmax(logits, -1)
                    
                    preds_prev[:, i+1] = pred[:, i]
                    preds[:, i] = pred[:, i]
                    i += 1
                    
                for nn, pp, ee in zip(n, preds, e): # sentence-wise
                    got = ''
                    for ppp in pp: # character-wise
                        if ppp == 1:
                            got += "*"
                        elif ppp == 2 or ppp == 0:
                            break
                        else: 
                            got += idx2surf.get(ppp, "*")
                    
                    error = distance.levenshtein(ee, got)
                    score = len(ee) - error
                    cum_score += score
                    full_score += len(ee)
                     
                    fout.write(u"{}\t{}\t{}\t{}\n".format(nn, ee, got, score))
            fout.write(u"Total acc.: {}/{}={}\n".format(cum_score, 
                                                        full_score, 
                                                        round(float(cum_score)/full_score, 2)))
                                        
if __name__ == '__main__':
    main()
    print("Done")