#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from generator import Generator
from discriminator import Discriminator

class Model(object):
    def __init__(self, z_dim, batch_size, clip_threshold):

        self.input_size = 256
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.clip_threshold = clip_threshold
        
        # generator config
        gen_layer = [1024, 512, 512, 256, 256, 128, 3]
        gen_in_dim = int(self.input_size/2**(len(gen_layer) - 1))

        #discriminato config
        disc_layer = [3, 64, 256, 512]

        # -- generator -----
        self.gen = Generator([u'gen_reshape', u'gen_deconv'],
                             gen_in_dim, gen_layer)

        # -- discriminator --
        self.disc = Discriminator([u'disc_conv', u'disc_fc'], disc_layer)
        self.lr = 0.00005

        
    def set_model(self):
        # -- z -> gen_fig -> disc ---

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])

        gen_figs = self.gen.set_model(self.z, self.batch_size, True, False)
        g_loss = self.disc.set_model(gen_figs, True, False)
        self.g_obj = - tf.reduce_mean(g_loss) # minus corresponds to maximization

        self.train_gen  = tf.train.RMSPropOptimizer(self.lr).minimize(self.g_obj, var_list = self.gen.get_variables())
        
        # -- true_fig -> disc --------
        self.figs= tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3])        

        d_loss = self.disc.set_model(self.figs, True, True)

        self.d_obj = tf.reduce_mean(-d_loss + g_loss)

        self.train_disc = tf.train.RMSPropOptimizer(self.lr).minimize(self.d_obj, var_list = self.disc.get_variables())
        
        # -- clipping --------
        c = self.clip_threshold
        self.disc_clip = [_.assign(tf.clip_by_value(_, -c, c)) for _ in self.disc.get_variables()]
        
        # -- for figure generation -------
        self.gen_figs = self.gen.set_model(self.z, self.batch_size, False, True)
        
    def training_gen(self, sess, z_list):
        _, g_obj = sess.run([self.train_gen, self.g_obj],
                            feed_dict = {self.z: z_list})
        return g_obj
        
    def training_disc(self, sess, z_list, figs):
        # optimize
        _, d_obj = sess.run([self.train_disc, self.d_obj],
                            feed_dict = {self.z: z_list,
                              self.figs:figs})
        # clipping
        sess.run(self.disc_clip)
        return d_obj
    
    def gen_fig(self, sess, z):
        ret = sess.run(self.gen_figs,
                       feed_dict = {self.z: z})
        return ret

if __name__ == u'__main__':
    model = Model(z_dim = 30, batch_size = 10, clip_threshold = 0.01)
    model.set_model()
    
