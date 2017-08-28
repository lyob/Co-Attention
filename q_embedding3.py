import os
import sys
import json
import re
import collections
import argparse
import random

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.lookup import index_to_string_table_from_file
from tensorflow.contrib import rnn

# Target log path
logs_path = '/Users/Benjamin/Documents/Ben/ibm/tf/Attention/logs_q'
writer = tf.summary.FileWriter(logs_path)

# json file containing some questions and answers for the CLEVR dataset
question_dir = '/Users/Benjamin/Documents/Ben/ibm/tf/Attention/CLEVR/questions/train_questions_subset_000.json'

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
#

## embedding and dictionary for all (question and answer) words
def build_embeddings(embedding_file, vocab_file):
  model = json.load(open(embedding_file)) # this comes from the word2vec embeddings
  vocabulary = [word.strip() for word in open(vocab_file)] # this comes from list of vocab used in the questions

  embedding_dim = len(model[vocabulary[1]][0]) # this is the size of the word2vec embeddings (=400?)

  embedding = np.zeros((len(vocabulary), embedding_dim)) # this is a matrix of size vocab_size x vocab size of word2vec embedding 
  dictionary = {'<PAD>': 0} 
  reverse_dictionary = {0: '<PAD>'}
  for i, word in enumerate(vocabulary[1:]):
    vec = np.asarray(model[word]).reshape(1, embedding_dim)
    embedding[i+1] = vec # this moves the vector corresponding to the word in the datafile into embedding matrix  
    reverse_dictionary[i+1] = word
    dictionary[word] = i+1
  return embedding, dictionary, reverse_dictionary
#
def serialize_examples(question_file, dictionary):
## numericised questions and answer dictionary (for only answer words)

  data = json.load(open(question_file))
  questions = []
  answers = []
  # num_questions = len(data['questions']) # currently using num_questions instead for faster computation/testing. Replace with num_questions later.
  num_questions = 1000
  for i, entry in zip(range(num_questions), data['questions']):
    questions.append(re.split(r'\s+', re.sub(r'([;?])', r'', entry['question'].lower())))
    answers.append(entry['answer'].lower())
    if i%100 == 0:
      update_progress(i, num_questions, 'Parsing example')
  update_progress(num_questions, num_questions, 'Parsing example')

  print('Creating answer dictionary...', end=' ')
  examples = [[dictionary[word.lower()] for word in q] for q in questions]
  labels = []

  answer_dict = {}
  for a in answers:
    if a not in answer_dict:
      answer_dict[a] = len(answer_dict)
    labels.append(answer_dict[a])

  reverse_answer_dict = dict(zip(answer_dict.values(), answer_dict.keys()))

  with open(os.path.join('data', 'answers.json'), 'w') as fp:
    json.dump(reverse_answer_dict, fp)

  assert len(labels) == len(examples), "Num examples does not match num labels"
  print('Done.')
  # 

  ## this serialises our sequential data so that we can feed it into an LSTM later and get input/outputs at multiple time steps. It uses tf.train.SequenceExample, which is Tensorflow's protocol buffer 
  def make_example(question, answer):
    ex = tf.train.SequenceExample()
    # we can add features to the data that is non-sequential, such as length.
    ex.context.feature['length'].int64_list.value.append(len(question))
    ex.context.feature['answer'].int64_list.value.append(answer_dict[answer])
    # usually 'answer' is sequential but for us it comes at the end of our data
    
    # feature lists for the sequential features of our example:
    fl_tokens = ex.feature_lists.feature_list['words']
    for token in question:
      fl_tokens.feature.add().int64_list.value.append(dictionary[token.lower()])
    return ex

  # Now that we converted our data into tf.SequenceExample format, we need to write it into one or more TFRecord files. We can use tf.TFRecordReader to read examples from the file, and tf.
  writer = tf.python_io.TFRecordWriter(os.path.join('data','train_examples.tfrecords'))
  for i, (question, answer) in enumerate(zip(questions, answers)):
    ex = make_example(question, answer)
    writer.write(ex.SerializeToString())
    if i%100 == 0:
      update_progress(i, num_questions, 'Serializing example')
  update_progress(num_questions, num_questions, 'Serializing example')
  writer.close()
#

## We batch the questions in:
def generate_batch(_inputs, batch_size):
  with tf.name_scope('Batch_Generator'):
    return tf.train.batch(_inputs, batch_size, num_threads=1, capacity=10768, dynamic_pad=True)
  # dynamic_pad=True does padding of questions automatically
#

# every time we use _inputs (like above) we want it to incorporate all three features
def preprocess_example(_inputs):
  length, question, answer = _inputs
  return [length, question, answer]

## data feed for the model 
def input_pipeline(input_files, batch_size):
  with tf.name_scope('Input'):
    filename_queue = tf.train.string_input_producer(input_files, num_epochs=1000)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    ## we can now parse an example:
    # define how to parse the example...
    context_features = {
      'length': tf.FixedLenFeature([], dtype=tf.int64),
      'answer': tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
      'words': tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    # ...and then parse it:
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example, context_features, sequence_features)
    length = context_parsed['length']
    question = sequence_parsed['words']
    answer = context_parsed['answer']

    example = preprocess_example((length, question, answer))

    ## generate batch:
    lengths, questions, answers = generate_batch(example, batch_size=batch_size)

    return lengths, questions, answers
#

## model
def bilstm(_input, lengths, embeddings):
  ### some parameters ###
  n_hidden = 1024

  ## we now use our embedding matrix to embed to vector space x_t. We also need the embedding for some semantic meaning
  with tf.variable_scope('Embedding'):
    init_embedding = tf.constant_initializer(embeddings) ## init_embedding are constants that are equal to function input 'embeddings'
    embeddings = tf.get_variable('Embeddings', [94, 300], initializer=init_embedding) # get.variable makes a new variable or edits an existing one. here we're making a new variable called "embeddings" that is size 94x300 and has values of input 'embeddings'.
    out = tf.nn.embedding_lookup(embeddings, _input) ## looking up input id's in the embedding matrix
  #

  with tf.variable_scope('Bidirectional_LSTM'): 
    lstm_fw_cell = rnn.LSTMCell(n_hidden, forget_bias=1.0) # Forward direction cell
    lstm_bw_cell = rnn.LSTMCell(n_hidden, forget_bias=1.0) # Backward direction cell

    (output_fw, output_bw), ((_, output_state_fw), (_, output_state_bw)) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, out, sequence_length = lengths, dtype=tf.float32)

    # state of lstm is tuple of memory and hidden state.
    # output_states is a tuple of the fw and bw final states
    final_state = tf.concat([output_state_fw, output_state_bw], axis=1)

    # hidden states at each time step. this is what we want for seq co-attn
    q_feature_vector = tf.concat([output_fw, output_bw], axis=1)

  return final_state, q_feature_vector
#

## Having defined our model, we can now run and train it 
def run_model(embedding):

  ### some parameters ###
  learning_rate = 0.0001 #$
  steps = 5000001 #$ arbitrary
  batch_size = 64 #$ arbitrary. number of questions fed into bidir lstm at one epoch
  ### Network Parameters ###
  n_hidden = 1024 # hidden layer num of features (usually arbitrary, but in paper they say dim(bidir LSTM) is 1/2 of dim(visual feature vector v_n), which is 2048 for ResNet101 and 512 for VGGNet-16). 
  n_classes = 28 # =len(answer_dict) the total number of possible answers
  example_batch = input_pipeline(['data/train_examples.tfrecords'], batch_size)
  lengths, questions, answers = example_batch

  final_state, q_feature_vector = bilstm(questions, lengths, embedding)
  ## Fully connected layer
  with tf.variable_scope('mlp'):
    # Define weights
    ## we use get_variable instead of Variable because we are using dynamic LSTM/RNN
    weights = tf.get_variable('weights', [2*n_hidden, n_classes], initializer=tf.random_normal_initializer()) ## Hidden layer weights => 2*n_hidden because of forward + backward cells
    biases = tf.get_variable('biases', [n_classes], initializer=tf.random_normal_initializer())
    out = tf.nn.xw_plus_b(final_state, weights, biases)
  #

  ## determine answers, accuracy and loss, which we can use to calculate backprop 
  with tf.variable_scope('eval'):
    predictions = tf.argmax(tf.nn.softmax(out), axis=1)
    correct = tf.equal(predictions, answers)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=answers))
  #

  with tf.variable_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # we can also use other optimizers:
    # train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    # train_op = tf.train.GradientDescentOptimizer(1.0).minimize(cost)
  #

  ## The numbers Mason! What do they mean?!?
  with tf.name_scope('Text_Decode'):
    question_table = index_to_string_table_from_file(vocabulary_file='data/vocab.txt', name='Question_Table')
    answer_table = index_to_string_table_from_file(vocabulary_file='data/answers.txt', name='Answer_Table')
    question_strings = tf.expand_dims(tf.reduce_join(question_table.lookup(tf.slice(questions, [0,0], [5, -1])), axis=1, separator=' '), axis=1)
    answer_strings = tf.expand_dims(answer_table.lookup(tf.slice(answers, [0], [5])), axis=1)
    prediction_strings = tf.expand_dims(answer_table.lookup(tf.slice(predictions, [0], [5])), axis=1)
    labels = tf.constant(['Question', 'Answer', 'Prediction'], shape=[1, 3])
    qa_table = tf.concat([question_strings, answer_strings, prediction_strings], axis=1)
    qa_table = tf.concat([labels, qa_table], axis=0)
    print(qa_table)
    # qa_string = tf.string_join([qa_string, prediction_strings], separator='\r\nPredicted: ')
    tf.summary.text('Question', qa_table)
  #

  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  saver = tf.train.Saver()

  ## tensorflow wizardry. We run the session, the tables, and write it to logs. 
  with tf.Session() as sess:
    tf.tables_initializer().run()
    sess.run(init_op)
    summary_writer = tf.summary.FileWriter('logs/3', graph=sess.graph)
    summary_op = tf.summary.merge_all()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(steps):
      if i%100 == 0:
        _, summary, acc = sess.run([train_op, summary_op, accuracy])
        summary_writer.add_summary(summary, i)
        print('\rStep %d, Accuracy: %f' %(i, acc), end=''*10)
      else:
        sess.run(train_op)
        print('\rStep %d' %i, end=' '*20)
      if i%5000 == 0:
        saver.save(sess, 'logs/3/model.ckpt', global_step=i)

    coord.request_stop()
    coord.join(threads)
#

def main(embedding):
  run_model(embedding)
#

if __name__ == '__main__':
  embedding, dictionary, reverse_dictionary = build_embeddings('data/embeddings.json', 'data/vocab.tsv')
  serialize_examples(question_dir, dictionary)
  main(embedding)
#




"""
Some useful links:
Some of the undocumented tensorflow functions used here: https://goo.gl/Hy1vhN
#
Other tensorflow stuff:
- https://jhui.github.io/2017/03/08/TensorFlow-variable-sharing/
- https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
#
Bidirectional LSTM:
1. https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks
2. https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn
3. https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
4. https://stackoverflow.com/questions/42936717/bi-directional-lstm-for-variable-length-sequence-in-tensorflow
5. https://stackoverflow.com/documentation/tensorflow/4827/creating-rnn-lstm-and-bidirectional-rnn-lstms-with-tensorflow#t=201708181923287793247
#
Other useful stuff:
http://stackabuse.com/reading-and-writing-json-to-a-file-in-python/


"""