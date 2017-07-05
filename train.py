# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
www.github.com/kyubyong/neural_japanese_transliterator
'''

from __future__ import print_function
import tensorflow as tf

from tqdm import tqdm
from data_load import get_batch, load_vocab, load_train_data
from hyperparams import Hyperparams as hp
from utils import shift_by_one
from modules import embed
from networks import encode, decode
from data_load import *

roma2idx, idx2roma, surf2idx, idx2surf = load_vocab()

class Graph:
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch()
            else: # Evaluation
                self.x = tf.placeholder(tf.int32, shape=(None, hp.max_len,))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.max_len,))
            
            # Character Embedding for x
            self.enc = embed(self.x, len(roma2idx), hp.embed_size, scope="emb_x")
                
            # Encoder
            self.memory = encode(self.enc, is_training=is_training)
            
            # Character Embedding for decoder_inputs
            self.decoder_inputs = shift_by_one(self.y)
            self.dec = embed(self.decoder_inputs, len(surf2idx), hp.embed_size, scope="emb_decoder_inputs")
            
            # Decoder
            self.outputs = decode(self.dec, self.memory, len(surf2idx), is_training=is_training) # (N, T', hp.n_mels*hp.r)
            self.logprobs = tf.log(tf.nn.softmax(self.outputs)+1e-10) 
            self.preds = tf.arg_max(self.outputs, dimension=-1)
                
            if is_training: 
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.outputs) 
                self.istarget = tf.to_float(tf.not_equal(self.y, tf.zeros_like(self.y))) # masking
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))
               
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                   
                # Summary 
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()
         
def main():   
    g = Graph(); print("Training Graph loaded")
    
    with g.graph.as_default():
        # Training 
        sv = tf.train.Supervisor(logdir=hp.logdir,
                                 save_model_secs=0)
        with sv.managed_session() as sess:
            for epoch in range(1, hp.num_epochs+1): 
                if sv.should_stop(): break
                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(g.train_op)

                # Write checkpoint files at every epoch
                gs = sess.run(g.global_step) 
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

if __name__ == '__main__':
    main()
    print("Done")
