import os
import sys
import json
import re
import collections
import argparse
import random

from collections import OrderedDict

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import rnn

# Target log path
logs_path = '/Users/Benjamin/Documents/Ben/ibm/tf/Attention/logs_q'
writer = tf.summary.FileWriter(logs_path)

# json file containing some questions and answers for the CLEVR dataset
file_dir = '/Users/Benjamin/Documents/Ben/ibm/tf/Attention/CLEVR/questions/train_questions_subset_000.json'

##### Data Extraction #####
## provides visual bars for updates
def update_progress(current, total, task):
    bar_length = 50
    text = ''
    progress = float(current)/total
    num_dots = int(round(bar_length*progress))
    num_spaces = bar_length - num_dots
    if current == total:
        text = '\r\nDone.\r\n'
    else:
        text = ('\r[{}] {:.2f}% ' + task + ' {} of {}').format('.'*num_dots + ' '*num_spaces, progress*100, current, total)
    sys.stdout.write(text)
    sys.stdout.flush()  

## extracts useful information about the data
def build_dictionary(question_file):
  data = json.load(open(question_file))
  questions = []
  answers = []
  # num_questions = len(data['questions']) # currently using num_questions instead for faster computation/testing. Replace with num_questions later.
  num_questions = 500
  q_vocab = []
  a_vocab = []
  vocab = []

  for i, entry in zip(range(num_questions), data['questions']):
    questions.append(re.split(r'\s+', re.sub(r'([;?])', r'', entry['question'].lower())))
    answers.append(entry['answer'].lower())
    for q in questions:
      for word in q:
        word = re.sub(';','',word)
        q_vocab.append(word)
        vocab.append(word)
    for word in answers:
      word = re.sub(';','',word)
      a_vocab.append(word)
      vocab.append(word)
    if i%100 == 0:
      update_progress(i, num_questions, 'Parsing example')
  update_progress(num_questions, num_questions, 'Parsing example')
 
  # build dictionary for all words (both question and answer).
  count = collections.Counter(vocab).most_common()
  countq = collections.Counter(q_vocab).most_common()
  counta = collections.Counter(a_vocab).most_common()
  dictionary = OrderedDict()
  ans_dict = OrderedDict()
  for word, _ in countq:
    dictionary[word] = len(dictionary)+1
  for word, _ in counta:
    ans_dict[word] = len(ans_dict)+1
  max_q_length = len(max(questions,key=len))
  vocab_size = len(dictionary)
  ans_voc_size = len(ans_dict)

  dictionary.update({'<PAD>':0})
  dictionary.move_to_end('<PAD>', last=False)
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  rvs_ans_dict = dict(zip(ans_dict.values(), ans_dict.keys()))
  return dictionary, reverse_dictionary, ans_dict, rvs_ans_dict, questions, max_q_length, vocab_size, ans_voc_size, num_questions

dictionary, reverse_dictionary, ans_dict, rvs_ans_dict, questions, max_q_length, vocab_size, ans_voc_size, num_questions = build_dictionary(file_dir)
print("longest question length is", max_q_length) # 41 for 2000 questions in CLEVR
print("Size of vocab is", vocab_size) # 80 for 2000 questions
print("size of ans vocab is", ans_voc_size) # 26 for 2000 questions
print("Number of questions extracted is", num_questions) # determined by num_questions

##### Pre-processing #####
# turns question words into numbers as defined by dictionary
def word_to_number(questions, dict):
    qstn_num = []
    for q, entry in enumerate(questions):
      #print(entry)
      lst = []
      if type(entry) is list:
        for i, word in enumerate(entry):
          #print(word)
          lst.append(dict[word])
        qstn_num.append(lst)
      else:
        qstn_num.append(ans_dict[entry])
    return qstn_num
qstn = word_to_number(questions, dictionary)
ans = word_to_number(answers, ans_dict)
# displays matrix of numericised questions
for l in qstn[0:9]:
  print([*l])
for a in ans[0:9]:
  print(a)

# adds padding of zeros to the numericised question matrix so they are all the same length
# inspired by https://goo.gl/8XvVVf
def padding(numerical_q, max_q_length):
    n = len(numerical_q)
    x = np.zeros([n, max_q_length], dtype=np.int32) # matrix of dim (number of questions, max question length)
    for i, x_i in enumerate(x):
        x_i[:len(numerical_q[i])] = numerical_q[i]
    return x
matrix = padding(qstn, max_q_length)
print(matrix)
print('first question is:',matrix[0]) # remember that 0 represents <PAD> in dictionary

## various one-hot encoding methods
# turns numericised questions into onehot matrix for question number 'value'
def one_hot_v(vector, dim):
  onehotmatrix = np.eye(dim+1)[vector]
  return onehotmatrix
onehotq = one_hot_v(matrix[value], vocab_size)
onehota = one_hot_v(ans, ans_voc_size)

# turns matrix of numericised questions into onehot tensor
def one_hot_t(matrix, depth):
  # If input "indices" of tf.one_hot is a matrix with shape [batch, features], the output shape will be (batch x features x depth) if axis == -1
  # for our matrix, features = max_q_length, batch = len(questions), depth = vocab_size
  out = tf.one_hot(matrix, depth, on_value = 1, off_value = 0, axis = -1)
  return out
onehot = one_hot_t(matrix, vocab_size)

# Alternatively, turns each question vector/list into a onehot tensor
def one_hot_tq(question_vector, depth):
  # axis of -1 gives output of tensor (features x depth)
  onehottq = tf.one_hot(question_vector, depth, axis == -1)
  return onehottq

## generates data batch to feed into the model. 
# to be used with one_hot_m
def generate_batchm(inputs, batch_size):
  return 

# to be used with one_hot_t
def generate_batch(inputs, batch_size):
  with tf.name_scope('Batch_Generator'):
    return tf.train.batch(inputs, batch_size, num_threads=1, capacity=10768, dynamic_pad=True)

##### Model #####
# This is inspired by https://goo.gl/W31vtv (static bidir RNN), https://goo.gl/9Dd5s4 (static RNN), https://goo.gl/ujqRvn (LSTM), https://goo.gl/fzHQiS (another RNN), and lastly Mikyas' CLEVR_BoW code.
def reset_graph():
    if 'sess' in globals() and sess:
    sess.close()
    tf.reset_default_graph()

def BiLSTM(x, vocab_size, n_hidden, embeddings):
  # Prepare data shape to match `bidirectional_rnn` function requirements
  # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
  # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
  #x = tf.unstack(x, n_steps, 1)
  with tf.variable_scope('Embedding'):
    init_embedding = tf.constant_initializer(embeddings):
    embeddings = tf.get_variable('embedding_matrix', [vocab_size, n_hidden])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

  with tf.variable_scope('BiLSTM'):
    # Define lstm cells with tensorflow
    lstm_fw_cell = rnn.LSTMCell(n_hidden, forget_bias=1.0) # Forward direction cell
    lstm_bw_cell = rnn.LSTMCell(n_hidden, forget_bias=1.0) # Backward direction cell
    # Get lstm cell output
    outputs, output_state_fw, output_state_bw = rnn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, sequence_length = max_q_length, dtype=tf.float32)
    # question feature vector is the concatenation of final fw and bw states
    q_feature_vector = tf.concat[output_state_fw, output_state_bw]
  return outputs, q_feature_vector

def train(onehot, embedding):
  ### Parameters
  learning_rate = 0.001
  num_epochs = 1000 # arbitrary. 
  batch_size = 100 # arbitrary. number of questions fed into bidir lstm at one epoch
  display_step = 1000
  ### Network Parameters
  vocab_size = vocab_size # self-explanatory, really
  n_input = max_q_length # data input length (generate a n_input-element sequence of inputs)
  n_hidden = 1024
  # hidden layer num of features (usually arbitrary, but in paper they say dim(bidir LSTM) is 1/2 of dim(visual feature vector v_n), which is 2048 for ResNet101 and 512 for VGGNet-16). 
  n_classes = vocab_size # the total number of possible answers, here it is the entire answer vocab. 

  graph = tf.Graph()
  config = projector.ProjectorConfig()

  batch = generate_batch(onehot, batch_size)
  batch = batch.reshape((batch_size, n_input))
  # x = tf.placeholder(tf.float32, [None, n_input, N])
  # y = tf.placeholder(tf.float32, [None, n_classes])

  # 1 layer Bi directional LSTM
  final_state, q_feature_vector = BiLSTM(batch, vocab_size, n_hidden, embedding)

  with tf.variable_scope('mlp'):
    # Define weights
    weights = {
      # Hidden layer weights => 2*n_hidden because of forward + backward cells
      'W': tf.get_variable(tf.random_normal([2*n_hidden, n_classes]))
    }
    biases = {
      'b': tf.get_variable(tf.random_normal([n_classes]))
    } 
    out = tf.matmul(final_state, weights['out']) + biases['out'])
    # out = tf.xw_plus_b(final_state, weights, biases)

  with tf.variable_scope('evaluation'):
    pred = tf.argmax(out, 1)
    answers = tf.argmax(y,1)
    correct = tf.equal(pred, answers)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))

  # Define loss and optimizer
  with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # we can also use other optimizers:
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cost)

  with tf.name_scope('Text Decode'):
    # look at larry's code 


  embedding = config.embeddings.add()
  embedding.tensor_name = embeddings.name 
  # should we use Word2Vec embeddings, or should we make this a weight as well? 
  # probably Word2Vec. Makes no sense to use only 1 hot

  # Initializing the variables
  init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  saver = tf.train.Saver()

  # begin training:
  with tf.Session(graph=graph) as sess:
    tf.tables_initializer().run()
    sess.run(init)
    print('Initialized')
    #summary_writer = tf.summary.FileWriter('logs/6', graph=sess.graph)
    writer.add_graph(graph=sess.graph)
    summary_op = tf.summary.merge_all()
    #projector.visualize_embeddings(summary_writer, config)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  for i in range():
    if i%100 == 0:
      _, summary, acc = sess.run([train_op, summary_op, accuracy])
      summary_writer.add_summary(summary, i)
      print('\rStep %d, Accuracy: %f' %(i, acc), end=''*10)
    else:
      sess.run(train_op)
      print('\rStep %d' %i, end=' '*20)
    if i%5000 == 0:
      saver.save(sess, 'logs/6/model.ckpt', global_step=i)
      
  coord.request_stop()
  coord.join(threads)

if __name__ == '__main__':
  embedding, dictionary, reverse_dictionary = build_embeddings('data/embeddings.json'
  serialize_examples(question_dir, dictionary)