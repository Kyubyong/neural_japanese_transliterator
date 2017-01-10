#-*- coding: utf-8 -*-
#!/usr/bin/python2
'''
Training.
'''
from __future__ import print_function
from prepro import Hyperparams, load_vocab, load_train_data
import sugartensor as tf

def get_batch_data():
    # Load data
    X, Y = load_train_data()
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([tf.convert_to_tensor(X, tf.int64), 
                                                  tf.convert_to_tensor(Y, tf.int64)])

    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                  num_threads=8,
                                  batch_size=Hyperparams.batch_size, 
                                  capacity=Hyperparams.batch_size*64,
                                  min_after_dequeue=Hyperparams.batch_size*32, 
                                  allow_smaller_final_batch=False) 
    num_batch = len(X) // Hyperparams.batch_size
    
    return x, y, num_batch # (64, 50) int64, (64, 50) int64, 1636

class ModelGraph():
    '''Builds a model graph'''
    def __init__(self, mode="train"):
        if mode == "train":
            self.x, self.y, self.num_batch = get_batch_data() # (64, 50) int64, (64, 50) int64, 1636
            self.y_src = tf.concat(1, [tf.zeros((Hyperparams.batch_size, 1), tf.int64), self.y[:, :-1]])
        else: # test
            self.x = tf.placeholder(tf.int64, [None, Hyperparams.seqlen])
            self.y_src = tf.placeholder(tf.int64, [None, Hyperparams.seqlen])

        self.roma2idx, _, self.surf2idx, _ = load_vocab()
        
        self.emb_x = tf.sg_emb(name='emb_x', voca_size=len(self.roma2idx), dim=Hyperparams.embed_dim)
        self.emb_y = tf.sg_emb(name='emb_y', voca_size=len(self.surf2idx), dim=Hyperparams.embed_dim)
        
        self.enc = self.x.sg_lookup(emb=self.emb_x)
        
        with tf.sg_context(size=5, act='relu', bn=True):
            for _ in range(20):
                dim = self.enc.get_shape().as_list()[-1]
                self.enc += self.enc.sg_conv1d(dim=dim) # (64, 50, 300) float32

        self.enc = self.enc.sg_concat(target=self.y_src.sg_lookup(emb=self.emb_y))
        
        self.dec = self.enc
        with tf.sg_context(size=5, act='relu', bn=True):
            for _ in range(20):
                dim = self.dec.get_shape().as_list()[-1]
                self._dec = tf.pad(self.dec, [[0, 0], [4, 0], [0, 0]])  # zero prepadding
                self.dec += self._dec.sg_conv1d(dim=dim, pad='VALID')  
                        
        # final fully convolutional layer for softmax
        self.logits = self.dec.sg_conv1d(size=1, dim=len(self.surf2idx), act='linear', bn=False) # (64, 50, 5072) float32
        if mode == "train":
            self.ce = self.logits.sg_ce(target=self.y, mask=False, one_hot=False) # (64, 50) float32
            self.istarget = tf.not_equal(self.y, tf.zeros_like(self.y)).sg_float() # (64, 50) float32
            self.reduced_loss = (self.ce * self.istarget).sg_sum() / (self.istarget.sg_sum() + 1e-5)
            tf.sg_summary_loss(self.reduced_loss, "reduced_loss")
            
def train():
    g = ModelGraph(); print("Graph loaded!")
    
    tf.sg_train(lr=0.0001, lr_reset=True, log_interval=10, loss=g.reduced_loss, max_ep=20, 
                save_dir='asset/train', early_stop=False, max_keep=10, ep_size=g.num_batch)
     
if __name__ == '__main__':
    train(); print("Done")