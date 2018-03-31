# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

__author__ = 'Jheng-Long Wu'

import sys
import os
import sys
import time
import argparse
import datetime
import numpy as np
import pickle as pk
import tensorflow as tf

import data_helpers
from text_nn import TextCNNFNN as textNN

from IPython import embed

print ('Command:', ' '.join(sys.argv), '\n')

parser = argparse.ArgumentParser()

# Data set path
parser.add_argument('--dataset_path', type=str, default='data/small', help='data set path')

# mention pair key
parser.add_argument('--train_mention_pair_key_coref', type=str, default='train_coref_mention_pair_key', help='key of coreference train file.')
parser.add_argument('--train_mention_pair_key_ncoref', type=str, default='train_ncoref_mention_pair_key', help='key of non-coreference train file.')
parser.add_argument('--development_mention_pair_key_coref', type=str, default='development_coref_mention_pair_key', help='key of coreference val file.')
parser.add_argument('--development_mention_pair_key_ncoref', type=str, default='development_ncoref_mention_pair_key', help='key of non-coreference val file.')
parser.add_argument('--test_mention_pair_key_coref', type=str, default='test_coref_mention_pair_key', help='key of coreference test file.')
parser.add_argument('--test_mention_pair_key_ncoref', type=str, default='test_ncoref_mention_pair_key', help='key of non-coreference test file.')

# String data sets of mention
parser.add_argument('--train_string_coref', type=str, default='train_coref_features_str', help='mention string of coreference train file.')
parser.add_argument('--train_string_ncoref', type=str, default='train_ncoref_features_str', help='mention string of non-coreference train file.')
parser.add_argument('--development_string_coref', type=str, default='development_coref_features_str', help='mention string of coreference val file.')
parser.add_argument('--development_string_ncoref', type=str, default='development_ncoref_features_str', help='mention string of non-coreference val file.')
parser.add_argument('--test_string_coref', type=str, default='test_coref_features_str', help='mention string of coreference test file.')
parser.add_argument('--test_string_ncoref', type=str, default='test_ncoref_features_str', help='mention string of non-coreference test file.')

# String data sets of addition
parser.add_argument('--train_string_add_coref', type=str, default='train_coref_features_str_add', help='mention dependency word dep of coreference train file.')
parser.add_argument('--train_string_add_ncoref', type=str, default='train_ncoref_features_str_add', help='mention dependency word dep of non-coreference train file.')
parser.add_argument('--development_string_add_coref', type=str, default='development_coref_features_str_add', help='mention dependency word dep of coreference val file.')
parser.add_argument('--development_string_add_ncoref', type=str, default='development_ncoref_features_str_add', help='mention dependency word dep of non-coreference val file.')
parser.add_argument('--test_string_add_coref', type=str, default='test_coref_features_str_add', help='mention dependency word dep of coreference test file.')
parser.add_argument('--test_string_add_ncoref', type=str, default='test_ncoref_features_str_add', help='mention dependency word dep of non-coreference test file.')

# String data sets of sentence
parser.add_argument('--train_string_sents_coref', type=str, default='train_coref_features_str_sents', help='sentence of mention of non-coreference train file.')
parser.add_argument('--train_string_sents_ncoref', type=str, default='train_ncoref_features_str_sents', help='sentence of mention of non-coreference train file.')
parser.add_argument('--development_string_sents_coref', type=str, default='development_coref_features_str_sents', help='sentence of mention of non-coreference val file.')
parser.add_argument('--development_string_sents_ncoref', type=str, default='development_ncoref_features_str_sents', help='sentence of mention of non-coreference val file.')
parser.add_argument('--test_string_sents_coref', type=str, default='test_coref_features_str_sents', help='sentence of mention of non-coreference test file.')
parser.add_argument('--test_string_sents_ncoref', type=str, default='test_ncoref_features_str_sents', help='sentence of mention of non-coreference test file.')

# Numeric data sets of mention
parser.add_argument('--train_numeric_coref', type=str, default='train_coref_features_num', help='mention numeric of coreference train file.')
parser.add_argument('--train_numeric_ncoref', type=str, default='train_ncoref_features_num', help='mention numeric of non-coreference train file.')
parser.add_argument('--development_numeric_coref', type=str, default='development_coref_features_num', help='mention numeric of coreference val file.')
parser.add_argument('--development_numeric_ncoref', type=str, default='development_ncoref_features_num', help='mention numeric of non-coreference val file.')
parser.add_argument('--test_numeric_coref', type=str, default='test_coref_features_num', help='mention numeric of coreference test file.')
parser.add_argument('--test_numeric_ncoref', type=str, default='test_ncoref_features_num', help='mention numeric of non-coreference test file.')

# Word embedding file
parser.add_argument('--pretrained_embedding', type=str, default=None, help='Pre-trained word embeddings file (only n by m matrix).')
parser.add_argument('--embedding_dim', type=int, default=300, help='Dimensionality of character embedding')

# Model Hyperparameters
parser.add_argument('--filter_sizes', type=str, default='2', help='Comma-separated filter sizes (i.e.: 3,4,5)')
parser.add_argument('--num_filters', type=int, default=200, help='Number of filters per filter size (default: 200)')
parser.add_argument('--num_dim_hidden', type=int, default=5, help='Number of hidden layer size (default: 200)')
parser.add_argument('--num_cnn_layers', type=int, default=5, help='Number of cnn layers')
parser.add_argument('--num_fnn_layers', type=int, default=10, help='Number of fnn layers')
parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help='Dropout keep probability (default: 0.5)')
parser.add_argument('--l2_reg_lambda', type=float, default=0.0, help='L2 regularizaion lambda (default: 0.0)')
parser.add_argument('--inital_learining_rate', type=float, default=1e-4, help='learning rate')

# Training parameters
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size (default: 128)')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs (default: 5)')
parser.add_argument('--checkpoint_every', type=int, default=100, help='Save model after this many steps (default: 100)')
parser.add_argument('--evaluate_every', type=int, default=1, help='Evaluate model after this many steps (default: 100)')
parser.add_argument('--used_mention', type=bool, default=True, help='use mention string information for training?')
parser.add_argument('--used_sentence', type=bool, default=True, help='use sentence string information for training?')
parser.add_argument('--used_addition', type=bool, default=True, help='use addition string information for training?')
parser.add_argument('--used_numeric', type=bool, default=True, help='use numeric information for training?')

# Misc Parameters
parser.add_argument('--allow_soft_placement', type=bool, default=True, help='Allow device soft device placement')
parser.add_argument('--log_device_placement', type=bool, default=False, help='Log placement of ops on devices')
parser.add_argument('--gpu_memory', type=float, default=0.3, help='How many precetage of GPU memory to used')

# Sampling for development and test dataset
parser.add_argument('--num_smaple', type=int, default=500, help='Number of samples per sampling size (default: 500)')

# run type
parser.add_argument('--checkpoint', type=str, default='runs/', help='Which check point you want to used.')
parser.add_argument('--restore', type=bool, default=False, help='Restore trained model')

FLAGS, unparsed = parser.parse_known_args()


if FLAGS.restore == False:
    print("Loading data:")
    if os.path.exists(os.path.join(FLAGS.dataset_path, 'balance')):
        data_length = pk.load(open(os.path.join(FLAGS.dataset_path, 'balance', 'data_length'),'rb'))
        vocabulary, vocabulary_inv = pk.load(open(os.path.join(FLAGS.dataset_path, 'balance', 'vocabulary'),'rb'))
    else:
        print ("runs data preprocessing")
        vocabulary, vocabulary_inv, data_length = data_helpers.load_data(FLAGS)
    print("Vocabulary Size: {:d}".format(len(vocabulary)))
    FLAGS.sequence_length_ment1 = data_length['sequence_length_ment1']
    FLAGS.sequence_length_ment2 = data_length['sequence_length_ment2']
    FLAGS.sequence_length_sents_ment1 = data_length['sequence_length_sents_ment1']
    FLAGS.sequence_length_sents_ment2 = data_length['sequence_length_sents_ment2']
    FLAGS.sequence_length_add_ment1 = data_length['sequence_length_add_ment1']
    FLAGS.sequence_length_add_ment2 = data_length['sequence_length_add_ment2']
    FLAGS.sequence_length_numeric_ment1 = data_length['sequence_length_numeric_ment1']
    FLAGS.sequence_length_numeric_ment2 = data_length['sequence_length_numeric_ment2']
    FLAGS.sequence_length_numeric = data_length['sequence_length_numeric']

else:
    # restore model
    print('!'*100)
    temp_num_epochs = FLAGS.num_epochs
    temp = FLAGS.checkpoint
    restore = FLAGS.restore
    FLAGS = pk.load(open(FLAGS.checkpoint+"/FLAGS", "rb"))
    FLAGS.checkpoint = temp
    FLAGS.restore = restore
    starting_epoch = FLAGS.num_epochs
    FLAGS.num_epochs = temp_num_epochs
    vocabulary = pk.load(open(FLAGS.checkpoint+"/vocabulary",'rb'))
    vocabulary_inv = pk.load(open(FLAGS.checkpoint+"/vocabulary_inv","rb"))

print("\nparameters:")
for attr, value in sorted(FLAGS.__dict__.items()):
    print ("  {} = {}".format(attr, value))

# Load embeddings
pretrained_embedding, FLAGS.embedding_dim = data_helpers.load_embedding(vocabulary, FLAGS.pretrained_embedding, FLAGS.embedding_dim)
print ("  dim. of word vector by setting", FLAGS.embedding_dim)

# Training
session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                              log_device_placement=FLAGS.log_device_placement,
                              gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory))
with tf.Session(config=session_conf) as sess:
    cnn = textNN(sequence_length_ment1=FLAGS.sequence_length_ment1,
                sequence_length_ment2=FLAGS.sequence_length_ment2,
                sequence_length_sents_ment1=FLAGS.sequence_length_sents_ment1,
                sequence_length_sents_ment2=FLAGS.sequence_length_sents_ment2,
                sequence_length_add_ment1=FLAGS.sequence_length_add_ment1,
                sequence_length_add_ment2=FLAGS.sequence_length_add_ment2,
                sequence_length_numeric_ment1=FLAGS.sequence_length_numeric_ment1,
                sequence_length_numeric_ment2=FLAGS.sequence_length_numeric_ment2,
                sequence_length_numeric=FLAGS.sequence_length_numeric,
                num_classes=2,
                pretrained_embedding=pretrained_embedding,
                embedding_size=FLAGS.embedding_dim,
                vocab_size=len(vocabulary),
                filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
                num_filters=FLAGS.num_filters,
                num_dim_hidden=FLAGS.num_dim_hidden,
                num_cnn_layers=FLAGS.num_cnn_layers,
                num_fnn_layers=FLAGS.num_fnn_layers,
                used_mention=FLAGS.used_mention,            
                used_sentence=FLAGS.used_sentence,
                used_addition=FLAGS.used_addition,
                used_numeric=FLAGS.used_numeric,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
    if FLAGS.restore == True:
        checkpoint_path = FLAGS.checkpoint + "/checkpoints"
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print ("The model has been restore from {} and traind global step {}".format(checkpoint_path, gs)) 
        print ("continue train model from epoch {} to {}".format(starting_epoch+1, FLAGS.num_epochs))
    else:
        gs = 0        
        starting_epoch = 0
    
    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    lr = tf.train.exponential_decay(
                            FLAGS.inital_learining_rate,
                            global_step,
                            10000,
                            0.95, 
                            staircase=True)
    
    optimizer = tf.train.AdamOptimizer(FLAGS.inital_learining_rate)
    
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir)

    # Test summaries
    test_summary_op = tf.summary.merge([loss_summary, acc_summary])
    test_summary_dir = os.path.join(out_dir, "summaries", "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    FLAGS.checkpoints = checkpoint_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables())
    
    # Dump the sequence length, vocabulary, vocabulary_inv to output directory
    pk.dump(FLAGS, open(os.path.join(out_dir, "FLAGS"),"wb"))
    pk.dump(vocabulary, open(os.path.join(out_dir, "vocabulary"), 'wb'))
    pk.dump(vocabulary_inv, open(os.path.join(out_dir, "vocabulary_inv"), 'wb'))

    # Initialize all variables
    tf.global_variables_initializer().run()

    def train_step(x_batch_ment1, x_batch_ment2,
                     x_batch_sents_ment1, x_batch_sents_ment2,
                     x_batch_add_ment1, x_batch_add_ment2,
                     x_batch_numeric_ment1, x_batch_numeric_ment2, x_batch_numeric,
                     y_batch):
        feed_dict = {cnn.input_y: y_batch, cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
        if FLAGS.used_numeric == True:
            feed_dict[cnn.input_numeric_ment1_x] = x_batch_numeric_ment1
            feed_dict[cnn.input_numeric_ment2_x] = x_batch_numeric_ment2
            feed_dict[cnn.input_numeric_x] = x_batch_numeric
        if FLAGS.used_mention == True:
            feed_dict[cnn.input_ment1_x] = x_batch_ment1
            feed_dict[cnn.input_ment2_x] = x_batch_ment2
        if FLAGS.used_sentence == True:
            feed_dict[cnn.input_sents_ment1_x] = x_batch_sents_ment1
            feed_dict[cnn.input_sents_ment2_x] = x_batch_sents_ment2
        if FLAGS.used_addition == True:
            feed_dict[cnn.input_add_ment1_x] = x_batch_add_ment1
            feed_dict[cnn.input_add_ment2_x] = x_batch_add_ment2

        _, step, summaries, loss, accuracy, scores, predictions, losses = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.scores, cnn.predictions, cnn.losses], feed_dict)

        time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d %H:%M:%S')
        print("{}: train step {} loss {:g} acc {:g}".format(time_str, step, loss, accuracy))

    def eval_step(x_batch_ment1, x_batch_ment2, x_batch_sents_ment1, x_batch_sents_ment2, x_batch_add_ment1, x_batch_add_ment2, x_batch_numeric_ment1, x_batch_numeric_ment2, x_batch_numeric, y_batch, sumary_op, writer=None):
        feed_dict = {cnn.input_y: y_batch, cnn.dropout_keep_prob: 1}
        if FLAGS.used_numeric == True:
            feed_dict[cnn.input_numeric_ment1_x] = x_batch_numeric_ment1
            feed_dict[cnn.input_numeric_ment2_x] = x_batch_numeric_ment2
            feed_dict[cnn.input_numeric_x] = x_batch_numeric
        if FLAGS.used_mention == True:
            feed_dict[cnn.input_ment1_x] = x_batch_ment1
            feed_dict[cnn.input_ment2_x] = x_batch_ment2
        if FLAGS.used_sentence == True:
            feed_dict[cnn.input_sents_ment1_x] = x_batch_sents_ment1
            feed_dict[cnn.input_sents_ment2_x] = x_batch_sents_ment2
        if FLAGS.used_addition == True:
            feed_dict[cnn.input_add_ment1_x] = x_batch_add_ment1
            feed_dict[cnn.input_add_ment2_x] = x_batch_add_ment2

        step, summaries, loss, accuracy, scores, predictions = sess.run(
            [global_step, sumary_op , cnn.loss, cnn.accuracy, cnn.scores, cnn.predictions],
            feed_dict)

        time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d %H:%M:%S')
        print("{}: eval loss {:g}, acc {:g}".format(time_str, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)
        return scores, predictions, accuracy
    # Load balance subsets statistics
    data_balance_sets = pk.load(open(os.path.join(FLAGS.dataset_path, 'balance', 'data_balance_sets'),'rb'))
    data_length = pk.load(open(os.path.join(FLAGS.dataset_path, 'balance', 'data_length'),'rb'))

    # train on training dataset
    print("[Runing Training stage]")
    for epoch in range(starting_epoch, FLAGS.num_epochs):
        # Load balance subsets statistics
        for set_num in range(data_balance_sets['train']):
            #if set_num == 1:
                #break
                #pass
            if set_num == 0:
                string_data, numeric_data, label_data = data_helpers.load_data_balance(FLAGS, 'train', set_num, data_length, vocabulary)
                y_train, x_train_ment1, x_train_ment2, x_train_sents_ment1, x_train_sents_ment2, x_train_add_ment1, x_train_add_ment2, x_train_numeric_ment1, x_train_numeric_ment2, x_train_numeric = data_helpers.load_subset(FLAGS, string_data, numeric_data, label_data, 'train')
            else:
                string_data, numeric_data, label_data = data_helpers.load_data_balance_ncoref(FLAGS, 'train', set_num, string_data, numeric_data, label_data, data_length, vocabulary)
                y_train, x_train_ment1, x_train_ment2, x_train_sents_ment1, x_train_sents_ment2, x_train_add_ment1, x_train_add_ment2, x_train_numeric_ment1, x_train_numeric_ment2, x_train_numeric = data_helpers.load_subset(FLAGS, string_data, numeric_data, label_data, 'train')
            print ("length on L/M/S/A/N/N:", len(y_train), len(x_train_ment1), len(x_train_sents_ment1), len(x_train_add_ment1), len(x_train_numeric_ment1), len(x_train_numeric))
            print ('length of coref. pair',  len(np.where(y_train[:,1]==1)[0]))
            # Generate batches
            zip_data = data_helpers.zip_features(x_train_ment1, x_train_ment2, x_train_sents_ment1, x_train_sents_ment2, x_train_add_ment1, x_train_add_ment2, x_train_numeric_ment1, x_train_numeric_ment2, x_train_numeric, y_train, FLAGS)
            batches = data_helpers.batch_iter_cnn_fnn(zip_data, FLAGS.batch_size)
            # Training loop. For each batch...
            for batch in batches:
                x_batch_ment1, x_batch_ment2, x_batch_sents_ment1, x_batch_sents_ment2, x_batch_add_ment1, x_batch_add_ment2, x_batch_numeric_ment1, x_batch_numeric_ment2, x_train_numeric, y_batch = data_helpers.unzip_features(batch, FLAGS)
                print ("subset {} epoch {} : ".format(set_num+1, epoch+1))
                train_step(x_batch_ment1, x_batch_ment2, x_batch_sents_ment1, x_batch_sents_ment2,
                 x_batch_add_ment1, x_batch_add_ment2, 
                 x_batch_numeric_ment1, x_batch_numeric_ment2, x_train_numeric, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            #break
    current_step = tf.train.global_step(sess, global_step)
    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
    print("Saved final model checkpoint to {}\n".format(path))

    # final evaluation on testing dataset
    print("[Runing evaluation stage]")
    scores = []
    predictions = []
    accuracies = []
    dataset_name = 'test'
    #data_balance_sets['test']
    for set_num in range(data_balance_sets[dataset_name]):
        print ("    subset of testing data on {}".format(set_num)),               
        if set_num == 0:
            string_data, numeric_data, label_data = data_helpers.load_data_balance(FLAGS, 'test', set_num, data_length, vocabulary)
            y_test, x_test_ment1, x_test_ment2, x_test_sents_ment1, x_test_sents_ment2, x_test_add_ment1, x_test_add_ment2, x_test_numeric_ment1, x_test_numeric_ment2, x_test_numeric = data_helpers.load_subset(FLAGS, string_data, numeric_data, label_data, 'test')
        else:
            string_data = {dataset_name:[[],[],[],[],[],[],[],[],[],[],[],[]]}
            numeric_data = {dataset_name:[[],[],[],[],[],[]]}
            label_data = {dataset_name:[[],[]]}
            string_data, numeric_data, label_data = data_helpers.load_data_balance_ncoref(FLAGS, 'test', set_num, string_data, numeric_data, label_data, data_length, vocabulary)
            y_test = label_data['test'][1]
            x_test_ment1 = string_data['test'][2]
            x_test_ment2 = string_data['test'][3]
            x_test_sents_ment1 = string_data['test'][6]
            x_test_sents_ment2 = string_data['test'][7]
            x_test_add_ment1 = string_data['test'][10]
            x_test_add_ment2 = string_data['test'][11]
            x_test_numeric_ment1 = numeric_data['test'][2]
            x_test_numeric_ment2 = numeric_data['test'][3]
            x_test_numeric = numeric_data['test'][5]
        test_sample_list, test_num_sets = data_helpers.data_smapling(y_test, FLAGS.num_smaple, rand=False)
        # Generate batches
        zip_data = data_helpers.zip_features(x_test_ment1, x_test_ment2, x_test_sents_ment1, x_test_sents_ment2, x_test_add_ment1, x_test_add_ment2, x_test_numeric_ment1, x_test_numeric_ment2, x_test_numeric, y_test, FLAGS)
        batches = data_helpers.batch_iter_cnn_fnn(zip_data, FLAGS.num_smaple)
        
        for batch in batches:
            x_batch_ment1, x_batch_ment2, x_batch_sents_ment1, x_batch_sents_ment2, x_batch_add_ment1, x_batch_add_ment2, x_batch_numeric_ment1, x_batch_numeric_ment2, x_train_numeric, y_batch = data_helpers.unzip_features(batch, FLAGS)
            s, p, a = eval_step(x_batch_ment1, x_batch_ment2, x_batch_sents_ment1, x_batch_sents_ment2, x_batch_add_ment1, x_batch_add_ment2,
                x_batch_numeric_ment1, x_batch_numeric_ment2, x_train_numeric, y_batch, test_summary_op, writer=test_summary_writer)
            scores.append(s)
            predictions.append(p)
            accuracies.append(a)
    print ("The average accuracy on testing subsets is {}".format(np.mean(accuracies)))
    print ("\n===The CNN model for coreference solution are finished===")