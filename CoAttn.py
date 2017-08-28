import os
import sys
import argparse
import collections
import json
import re

import tensorflow as tf
import numpy as np

### Sequential Co-Attention ###

def preprocess(layer):
  v_n = tf.concat(tf.unstack(layer, 14, axis=1), axis=0)
  return v_n

def base_vector(_input, shape): # we use get_variable with Wm beacuse it is shared between the visul and question attention streams
  Wm = tf.get_variable("Wm", shape, initializer=tf.random_normal_initializer())
  base_vector = tf.nn.tanh(tf.matmul(_input, Wm)) 
  return base_vector

def softmax(_input, shape)
  with tf.variable_scope('softmax'):
    Wh = tf.get_variable("Wh", shape, initializer=tf.random_normal_initializer())
    model = tf.nn.softmax(tf.matmul(_input, Wh))
  return softmax

def coattention(q_t, v_n, lengths, N=196, dim=2048, wdim=2048): # inputs are q_feature_vector, image feature vector, T, N, dimension of vectors, dim of weight matrices, arbitrarily set at 2048
  v_n = preprocess(last_pool)
  
  # splits the input tensors into time or feature dependent tensors: {q_t} --> q_t, {v_n} --> v_n
  q_t_list = []
  q_t_list = tf.unstack(q_t, lengths, axis=0) # but this axis might also be 1
  q_0 = tf.scalar_mul(np.divide(1,scalar), tf.add_n(q_t)) # tensor representing the question, combined into one 1x2048 tensor 
  
  v_n_list = []
  v_n_list = tf.unstack(v_n, N, axis=1)
  v_0 = tf.tanh(tf.scalar_mul(np.divide(1,N), tf.add_n(v_n))) # tensor representing the image, combined into one 1x2048 tensor

  m_0 = tf.multiply(q_0,v_0) # element-wise multiplication, 1x2048 tensor

  with tf.variable_scope('visual_attn', reuse = True): # reuse=True means we share any get_variable that's in the scope
    with tf.variable_scope('v_layer_1'):
      Wv = tf.Variable("Wv", [dim, dimw], initializer=tf.random_normal_initializer()) # weight for visual feature vector
      for i in v_n_list:
        feature_vector_v = tf.tanh(tf.matmul(Wv, v_n_list[i]))
        hidden_n = tf.multiply(feature_vector_v, base_vector(m_0, [dim, dimw])) # the numbers here are arbitrary until I know what the dim of the weights should be
    with tf.variable_scope('v_layer_2'):
      alpha_n = softmax(hidden_n, [dim, wdim])
      v_star = tf.tanh(tf.add_n(tf.matmul(alpha_n, v_n)))

  with tf.variable_scope('question_attn', reuse = True):
    with tf.variable_scope('q_layer_1'):
      Wq = tf.Variable("Wq", [dim, dimw], initializer=tf.random_normal_initializer())
      feature_vector_q = tf.tanh(tf.matmul(Wq, q_t))
      hidden_t = tf.multiply(feature_vector_q, base_vector(m_0, [dim, dimw]))
    with tf.variable_scope('q_layer_2'):
      alpha_t = softmax(hidden_n, [dim, wdim])
      q_star = tf.add_n(tf.matmul(alpha_t, q_t))

  with tf.variable_scope('coattn_output'):
    x_t = tf.concat(v_star, q_star)

  return x_t
#

def run_coattn():
  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  with tf.Session() as sess:
    run.sess(init_op)

def main():
