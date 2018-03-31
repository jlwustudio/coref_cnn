# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

__author__ = 'Jheng-Long Wu'

import os
import re
import sys
import argparse
import datetime
import itertools
import numpy as np
import pickle as pk  
from collections import Counter

from IPython import embed

def split_word_process(x_data):    
    x_data = [clean_str(sent).split(" ") for sent in x_data]
    return x_data

def load_mention_pair_key(FLAGS):
    # Load data from files on mention pair keys
    print ("  Load datasets on mention pair key...")
    train_coref_mention_pair_key = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_mention_pair_key_coref)).readlines())
    train_ncoref_mention_pair_key = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_mention_pair_key_ncoref)).readlines())
    dev_coref_mention_pair_key = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_mention_pair_key_coref)).readlines())
    dev_ncoref_mention_pair_key = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_mention_pair_key_ncoref)).readlines())
    test_coref_mention_pair_key = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_mention_pair_key_coref)).readlines())
    test_ncoref_mention_pair_key = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_mention_pair_key_ncoref)).readlines())
    
    # splite by item
    print ("  Split by item for mention pair keys")
    train_coref_mention_pair_key = np.array([s.strip().split(" ") for s in train_coref_mention_pair_key])
    train_ncoref_mention_pair_key = np.array([s.strip().split(" ") for s in train_ncoref_mention_pair_key])
    dev_coref_mention_pair_key = np.array([s.strip().split(" ") for s in dev_coref_mention_pair_key])
    dev_ncoref_mention_pair_key = np.array([s.strip().split(" ") for s in dev_ncoref_mention_pair_key])
    test_coref_mention_pair_key = np.array([s.strip().split(" ") for s in test_coref_mention_pair_key])
    test_ncoref_mention_pair_key = np.array([s.strip().split(" ") for s in test_ncoref_mention_pair_key])
    mention_pair_key_data = {'train':[train_coref_mention_pair_key, train_ncoref_mention_pair_key],'dev':[dev_coref_mention_pair_key, dev_ncoref_mention_pair_key], 'test':[test_coref_mention_pair_key, test_ncoref_mention_pair_key]}
    return mention_pair_key_data

def load_string_data_and_labels(FLAGS):
    # Load data from files on mention
    print ("  Load train dataset on mention...")
    train_coref_examples_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_string_coref + '_ment1')).readlines())
    train_ncoref_examples_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_string_ncoref + '_ment1')).readlines())
    train_coref_examples_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_string_coref + '_ment2')).readlines())
    train_ncoref_examples_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_string_ncoref + '_ment2')).readlines())
    print ("  Load development dataset on mention")
    dev_coref_examples_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_string_coref + '_ment1')).readlines())
    dev_ncoref_examples_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_string_ncoref + '_ment1')).readlines())
    dev_coref_examples_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_string_coref + '_ment2')).readlines())
    dev_ncoref_examples_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_string_ncoref + '_ment2')).readlines())
    print ("  Load test dataset on mention")
    test_coref_examples_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_string_coref + '_ment1')).readlines())
    test_ncoref_examples_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_string_ncoref + '_ment1')).readlines())
    test_coref_examples_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_string_coref + '_ment2')).readlines())
    test_ncoref_examples_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_string_ncoref + '_ment2')).readlines())

    # Load data from files on sentence of mention
    print ("  Load train dataset on sentence...")
    train_coref_examples_sents_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_string_sents_coref + '_ment1')).readlines())
    train_ncoref_examples_sents_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_string_sents_ncoref + '_ment1')).readlines())
    train_coref_examples_sents_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_string_sents_coref + '_ment2')).readlines())
    train_ncoref_examples_sents_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_string_sents_ncoref + '_ment2')).readlines())
    print ("  Load development dataset on sentence")
    dev_coref_examples_sents_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_string_sents_coref + '_ment1')).readlines())
    dev_ncoref_examples_sents_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_string_sents_ncoref + '_ment1')).readlines())
    dev_coref_examples_sents_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_string_sents_coref + '_ment2')).readlines())
    dev_ncoref_examples_sents_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_string_sents_ncoref + '_ment2')).readlines())
    print ("  Load test dataset on sentence")
    test_coref_examples_sents_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_string_sents_coref + '_ment1')).readlines())
    test_ncoref_examples_sents_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_string_sents_ncoref + '_ment1')).readlines())
    test_coref_examples_sents_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_string_sents_coref + '_ment2')).readlines())
    test_ncoref_examples_sents_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_string_sents_ncoref + '_ment2')).readlines())

    # Load data from files on addition of mention
    print ("  Load train dataset on addition...")
    train_coref_examples_add_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_string_add_coref + '_ment1')).readlines())
    train_ncoref_examples_add_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_string_add_ncoref + '_ment1')).readlines())
    train_coref_examples_add_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_string_add_coref + '_ment2')).readlines())
    train_ncoref_examples_add_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_string_add_ncoref + '_ment2')).readlines())
    print ("  Load development dataset on addition")
    dev_coref_examples_add_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_string_add_coref + '_ment1')).readlines())
    dev_ncoref_examples_add_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_string_add_ncoref + '_ment1')).readlines())
    dev_coref_examples_add_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_string_add_coref + '_ment2')).readlines())
    dev_ncoref_examples_add_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_string_add_ncoref + '_ment2')).readlines())
    print ("  Load test dataset on addition")
    test_coref_examples_add_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_string_add_coref + '_ment1')).readlines())
    test_ncoref_examples_add_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_string_add_ncoref + '_ment1')).readlines())
    test_coref_examples_add_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_string_add_coref + '_ment2')).readlines())
    test_ncoref_examples_add_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_string_add_ncoref + '_ment2')).readlines())

    # Remove space char on mention
    print ("  Remove space char and split for train dataset on mention")
    train_coref_examples_ment1 = [clean_str(s.strip()).split(" ") for s in train_coref_examples_ment1]
    train_ncoref_examples_ment1 = [clean_str(s.strip()).split(" ") for s in train_ncoref_examples_ment1]
    train_coref_examples_ment2 = [clean_str(s.strip()).split(" ") for s in train_coref_examples_ment2]
    train_ncoref_examples_ment2 = [clean_str(s.strip()).split(" ") for s in train_ncoref_examples_ment2]
    
    print ("  Remove space char and split for development dataset on mention")
    dev_coref_examples_ment1 = [clean_str(s.strip()).split(" ") for s in dev_coref_examples_ment1]
    dev_ncoref_examples_ment1 = [clean_str(s.strip()).split(" ") for s in dev_ncoref_examples_ment1]
    dev_coref_examples_ment2 = [clean_str(s.strip()).split(" ") for s in dev_coref_examples_ment2]
    dev_ncoref_examples_ment2 = [clean_str(s.strip()).split(" ") for s in dev_ncoref_examples_ment2]
    print ("  Remove space char and split for test dataset on mention")
    test_coref_examples_ment1 = [clean_str(s.strip()).split(" ") for s in test_coref_examples_ment1]
    test_ncoref_examples_ment1 = [clean_str(s.strip()).split(" ") for s in test_ncoref_examples_ment1]
    test_coref_examples_ment2 = [clean_str(s.strip()).split(" ") for s in test_coref_examples_ment2]
    test_ncoref_examples_ment2 = [clean_str(s.strip()).split(" ") for s in test_ncoref_examples_ment2]

    # Remove space char on sentence of mention
    print ("  Remove space char and split for train dataset on sentence")
    train_coref_examples_sents_ment1 = [clean_str(s.strip()).split(" ") for s in train_coref_examples_sents_ment1]
    train_ncoref_examples_sents_ment1 = [clean_str(s.strip()).split(" ") for s in train_ncoref_examples_sents_ment1]
    train_coref_examples_sents_ment2 = [clean_str(s.strip()).split(" ") for s in train_coref_examples_sents_ment2]
    train_ncoref_examples_sents_ment2 = [clean_str(s.strip()).split(" ") for s in train_ncoref_examples_sents_ment2]
    print ("  Remove space char and split for development dataset on sentence")
    dev_coref_examples_sents_ment1 = [clean_str(s.strip()).split(" ") for s in dev_coref_examples_sents_ment1]
    dev_ncoref_examples_sents_ment1 = [clean_str(s.strip()).split(" ") for s in dev_ncoref_examples_sents_ment1]
    dev_coref_examples_sents_ment2 = [clean_str(s.strip()).split(" ") for s in dev_coref_examples_sents_ment2]
    dev_ncoref_examples_sents_ment2 = [clean_str(s.strip()).split(" ") for s in dev_ncoref_examples_sents_ment2]
    print ("  Remove space char and split for test dataset on sentence")
    test_coref_examples_sents_ment1 = [clean_str(s.strip()).split(" ") for s in test_coref_examples_sents_ment1]
    test_ncoref_examples_sents_ment1 = [clean_str(s.strip()).split(" ") for s in test_ncoref_examples_sents_ment1]
    test_coref_examples_sents_ment2 = [clean_str(s.strip()).split(" ") for s in test_coref_examples_sents_ment2]
    test_ncoref_examples_sents_ment2 = [clean_str(s.strip()).split(" ") for s in test_ncoref_examples_sents_ment2]
    
    # Remove space char on addition of mention
    print ("  Remove space char and split for train dataset on addition")
    train_coref_examples_add_ment1 = [clean_str(s.strip()).replace(" ", "_").split(" ") for s in train_coref_examples_add_ment1]
    train_ncoref_examples_add_ment1 = [clean_str(s.strip()).replace(" ", "_").split(" ") for s in train_ncoref_examples_add_ment1]
    train_coref_examples_add_ment2 = [clean_str(s.strip()).replace(" ", "_").split(" ") for s in train_coref_examples_add_ment2]
    train_ncoref_examples_add_ment2 = [clean_str(s.strip()).replace(" ", "_").split(" ") for s in train_ncoref_examples_add_ment2]
    print ("  Remove space char and split for development dataset on addition")
    dev_coref_examples_add_ment1 = [clean_str(s.strip()).replace(" ", "_").split(" ") for s in dev_coref_examples_add_ment1]
    dev_ncoref_examples_add_ment1 = [clean_str(s.strip()).replace(" ", "_").split(" ") for s in dev_ncoref_examples_add_ment1]
    dev_coref_examples_add_ment2 = [clean_str(s.strip()).replace(" ", "_").split(" ") for s in dev_coref_examples_add_ment2]
    dev_ncoref_examples_add_ment2 = [clean_str(s.strip()).replace(" ", "_").split(" ") for s in dev_ncoref_examples_add_ment2]
    print ("  Remove space char and split for test dataset on addition")
    test_coref_examples_add_ment1 = [clean_str(s.strip()).replace(" ", "_").split(" ") for s in test_coref_examples_add_ment1]
    test_ncoref_examples_add_ment1 = [clean_str(s.strip()).replace(" ", "_").split(" ") for s in test_ncoref_examples_add_ment1]
    test_coref_examples_add_ment2 = [clean_str(s.strip()).replace(" ", "_").split(" ") for s in test_coref_examples_add_ment2]
    test_ncoref_examples_add_ment2 = [clean_str(s.strip()).replace(" ", "_").split(" ") for s in test_ncoref_examples_add_ment2]

    # Generate labels for examples
    print ("  Generate labels for train dataset,\n    coref. {}, non-coref. {}".format(len(train_coref_examples_ment1), len(train_ncoref_examples_ment1)))
    one_hot_weight = 1
    train_coref_labels = np.array([[0, one_hot_weight] for _ in range(len(train_coref_examples_ment1))])
    train_ncoref_labels = np.array([[one_hot_weight, 0] for _ in range(len(train_ncoref_examples_ment1))])
    print ("  Generate labels for development dataset,\n    coref. {}, non-coref. {}".format(len(dev_coref_examples_ment1), len(dev_ncoref_examples_ment1)))
    dev_coref_labels = np.array([[0, one_hot_weight] for _ in range(len(dev_coref_examples_ment1))])
    dev_ncoref_labels = np.array([[one_hot_weight, 0] for _ in range(len(dev_ncoref_examples_ment1))])
    print ("  Generate labels for test dataset,\n    coref. {}, non-coref. {}".format(len(test_coref_examples_ment1), len(test_ncoref_examples_ment1)))
    test_coref_labels = np.array([[0, one_hot_weight] for _ in range(len(test_coref_examples_ment1))])
    test_ncoref_labels = np.array([[one_hot_weight, 0] for _ in range(len(test_ncoref_examples_ment1))])
    
    label_data = {'train':[train_coref_labels, train_ncoref_labels], 'dev':[dev_coref_labels,dev_ncoref_labels],'test':[test_coref_labels, test_ncoref_labels]}
    
    string_data = {}
    string_data['train'] = [train_coref_examples_ment1, train_coref_examples_ment2, train_ncoref_examples_ment1, train_ncoref_examples_ment2, train_coref_examples_sents_ment1, train_coref_examples_sents_ment2, train_ncoref_examples_sents_ment1, train_ncoref_examples_sents_ment2,
    train_coref_examples_add_ment1, train_coref_examples_add_ment2, train_ncoref_examples_add_ment1, train_ncoref_examples_add_ment2]
    string_data['dev'] = [dev_coref_examples_ment1, dev_coref_examples_ment2, dev_ncoref_examples_ment1, dev_ncoref_examples_ment2, dev_coref_examples_sents_ment1, dev_coref_examples_sents_ment2, dev_ncoref_examples_sents_ment1, dev_ncoref_examples_sents_ment2,
    dev_coref_examples_add_ment1, dev_coref_examples_add_ment2, dev_ncoref_examples_add_ment1, dev_ncoref_examples_add_ment2]
    string_data['test'] = [test_coref_examples_ment1, test_coref_examples_ment2, test_ncoref_examples_ment1, test_ncoref_examples_ment2, test_coref_examples_sents_ment1, test_coref_examples_sents_ment2, test_ncoref_examples_sents_ment1, test_ncoref_examples_sents_ment2,
    test_coref_examples_add_ment1, test_coref_examples_add_ment2, test_ncoref_examples_add_ment1, test_ncoref_examples_add_ment2]
    
    return string_data, label_data

def pad_sentences_training(data, sequence_length, vocabulary, padding_word="<PAD/>"):
    padding_index = vocabulary[padding_word]
    for sentences in data:
        for i in range(len(sentences)):
            num_padding = sequence_length - len(sentences[i])
            if num_padding != 0:
                sentences[i] = np.array(sentences[i] + [padding_index] * num_padding)
            
def pad_sentences(data, sequence_length, padding_word="<PAD/>"):
    for sentences in data:
        for i in range(len(sentences)):
            num_padding = sequence_length - len(sentences[i])
            sentences[i] = sentences[i] + [padding_word] * num_padding
   
def pad_sentences_testing(sentences, sequence_length, padding_word="<PAD/>"):
    # only for tsting stage
    for i in range(len(sentences)):
        if len(sentences[i]) > sequence_length:
            sentences[i] = sentences[i][0:sequence_length]
        num_padding = sequence_length - len(sentences[i])
        sentences[i] = sentences[i] + [padding_word] * num_padding
    return sentences

def build_vocab(sentences, nonappear_word="<NAW/>"):
    print ("    Build vocabulary")
    sentences.append(["<PAD/>"])
    sentences.append([nonappear_word]*len(sentences[0]))
    word_counts = Counter(list(itertools.chain(*sentences)))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(string_dataset, vocabulary, nonappear_word="<NAW/>"):
    for dataset in range(len(string_dataset)):
        for sentence in range(len(string_dataset[dataset])):
            for word in range(len(string_dataset[dataset][sentence])):
                string_dataset[dataset][sentence][word] = vocabulary[string_dataset[dataset][sentence][word]]
        string_dataset[dataset] = np.array(string_dataset[dataset])

def load_embedding(vocabs, embedding_file, embedding_dim, nonappear_word="<NAW/>"):
    embeddings = None
    if embedding_file == None:
        return embeddings, embedding_dim
    print ("Loading pre-trained embeddings from {}".format(embedding_file))
    match = []
    pretrained_embeddings = list(open(embedding_file).readlines())
    embedding_dim = len(pretrained_embeddings[0].strip().split(" ")[1:])
    embeddings = [np.random.rand(embedding_dim).tolist()]*len(vocabs)
    for line in pretrained_embeddings:
        line = line.strip().split(" ")
        try:
            idx = vocabs[line[0]]
        except:
            idx = -1
        if idx == -1:
            continue
        embeddings[idx] = [float(l) for l in line[1:]]
        match.append(idx)
    embeddings[vocabs["<PAD/>"]] = [0]*embedding_dim
    print ("  numbef of vocabs not in pre-trained embeddings {}".format(len(embeddings)-len(match)))
    print ("  new embedding, vocabs {}, dim. {}".format(len(embeddings), embedding_dim))
    return embeddings, embedding_dim

def compute_sequence_length(string_data_ment):
    max_value = []
    for dataset in string_data_ment:
        max_value.append(max([len(x) for x in dataset]))
    return max(max_value)

def load_data(FLAGS):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    print (FLAGS.dataset_path)
    print ("Load mention pair keys")
    mention_pair_key_data = load_mention_pair_key(FLAGS)
    print ("Load numeric data:")
    numeric_data = load_numeric_data(FLAGS)
    print ("Load string data:")
    string_data, label_data = load_string_data_and_labels(FLAGS)
    

    print ("  Builds a vocabulary mapping from word to index based on the sentences")
    train_sentences = [s for d in string_data['train'] for s in d]
    dev_sentences = [s for d in string_data['dev'] for s in d]
    test_sentences = [s for d in string_data['test'] for s in d]
    vocabulary, vocabulary_inv = build_vocab(train_sentences + dev_sentences + test_sentences)
    print ("  total words in vocabulary {}".format(len(vocabulary)))

    sequence_length_ment1 = compute_sequence_length([string_data['train'][0], string_data['train'][2], string_data['dev'][0], string_data['dev'][2],string_data['test'][0],string_data['test'][2]])
    sequence_length_ment2 = compute_sequence_length([string_data['train'][1], string_data['train'][3], string_data['dev'][1], string_data['dev'][3],string_data['test'][1],string_data['test'][3]])
    sequence_length_sents_ment1 = compute_sequence_length([string_data['train'][4], string_data['train'][6], string_data['dev'][4], string_data['dev'][6],string_data['test'][4],string_data['test'][6]])
    sequence_length_sents_ment2 = compute_sequence_length([string_data['train'][5], string_data['train'][7], string_data['dev'][5], string_data['dev'][7],string_data['test'][5],string_data['test'][7]])

    sequence_length_add_ment1 = compute_sequence_length([string_data['train'][8], string_data['train'][10], string_data['dev'][8], string_data['dev'][10],string_data['test'][8],string_data['test'][10]])
    sequence_length_add_ment2 = compute_sequence_length([string_data['train'][9], string_data['train'][11], string_data['dev'][9], string_data['dev'][11],string_data['test'][9],string_data['test'][11]])
    
    sequence_length_numeric_ment1 = compute_sequence_length([numeric_data['train'][0], numeric_data['train'][2], numeric_data['dev'][0], numeric_data['dev'][0], numeric_data['test'][0], numeric_data['test'][0]])
    sequence_length_numeric_ment2 = compute_sequence_length([numeric_data['train'][1], numeric_data['train'][3], numeric_data['dev'][1], numeric_data['dev'][3], numeric_data['test'][1], numeric_data['test'][3]])
    sequence_length_numeric = compute_sequence_length([numeric_data['train'][4], numeric_data['train'][5], numeric_data['dev'][4], numeric_data['dev'][5], numeric_data['test'][4], numeric_data['test'][5]])
    print ("Sequence length: m1/m2/am1/am2/sm1/sm2/n1/n2/nr -> ", sequence_length_ment1, sequence_length_ment2, sequence_length_add_ment1, sequence_length_add_ment2, sequence_length_sents_ment1, sequence_length_sents_ment2,)
    print (sequence_length_numeric_ment1, sequence_length_numeric_ment2, sequence_length_numeric)

    data_length = {}
    data_length['sequence_length_ment1'] = sequence_length_ment1
    data_length['sequence_length_ment2'] = sequence_length_ment2
    data_length['sequence_length_sents_ment1'] = sequence_length_sents_ment1
    data_length['sequence_length_sents_ment2'] = sequence_length_sents_ment2
    data_length['sequence_length_add_ment1'] = sequence_length_add_ment1
    data_length['sequence_length_add_ment2'] = sequence_length_add_ment2
    data_length['sequence_length_numeric_ment1'] = sequence_length_numeric_ment1
    data_length['sequence_length_numeric_ment2'] = sequence_length_numeric_ment2
    data_length['sequence_length_numeric'] = sequence_length_numeric
   
    print ("Maps sentencs and labels of mention to vectors based on a vocabulary.")
    print ("  on train dataset")
    build_input_data(string_data['train'], vocabulary)    
    print ("  on developtment dataset")
    build_input_data(string_data['dev'], vocabulary)
    print ("  on test dataset")
    build_input_data(string_data['test'], vocabulary)

    # now, there has 5 datastts including string_data, numeric_data, label_data, vocabulary, vocabulary_inv
    # string_data is dic. that has {train, dev, test},
    #   each string_data is list, the sequence of type is 
    #     [coref_examples_ment1, coref_examples_ment2, ncoref_examples_ment1, ncoref_examples_ment2, coref_examples_sents_ment1, coref_examples_sents_ment2, ncoref_examples_sents_ment1, ncoref_examples_sents_ment2)
    # label_data is dic. that {train, dev, test}, 
    #   each label_data is list, the sequence of type is [coref_labels, ncoref_labels]
    # numeric_data is dic. that has {train, dev, test},
    #   each numeric_data is list, the sequence of type is [coref_examples_ment1, coref_examples_ment2, ncoref_examples_ment1, ncoref_examples_ment2]
    
    save_balacne_data(mention_pair_key_data, string_data, numeric_data, label_data, vocabulary, vocabulary_inv, data_length, FLAGS.dataset_path)
    return vocabulary, vocabulary_inv, data_length

def load_data_balance(FLAGS, dataset_name, set_num, data_length, vocabulary):
    balace_dir = os.path.join(FLAGS.dataset_path, 'balance')
    if os.path.exists(balace_dir):
        string_data = {dataset_name:[[],[],[],[],[],[],[],[],[],[],[],[]]}
        numeric_data = {dataset_name:[[],[],[],[],[],[]]}
        label_data = {dataset_name:[[],[]]}
        print ("    restore balanced data on {} from {}, subset {}".format(dataset_name, FLAGS.dataset_path, set_num),)
        label_data[dataset_name][0] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_label_coref'),'rb'))
        label_data[dataset_name][1] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_label_ncoref' + str(set_num)),'rb'))
        if FLAGS.used_mention == True:
            string_data[dataset_name][0], string_data[dataset_name][1] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_string_coref'),'rb'))
            string_data[dataset_name][2], string_data[dataset_name][3] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_string_ncoref' + str(set_num)),'rb'))
        if FLAGS.used_sentence == True:
            string_data[dataset_name][4], string_data[dataset_name][5] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_string_sent_coref'),'rb'))        
            string_data[dataset_name][6], string_data[dataset_name][7] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_string_sent_ncoref' + str(set_num)),'rb'))
        if FLAGS.used_addition == True:
            string_data[dataset_name][8], string_data[dataset_name][9] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_string_add_coref'),'rb'))        
            string_data[dataset_name][10], string_data[dataset_name][11] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_string_add_ncoref' + str(set_num)),'rb'))
        if FLAGS.used_numeric == True:
            numeric_data[dataset_name][0], numeric_data[dataset_name][1], numeric_data[dataset_name][4] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_numeric_coref'),'rb'))
            numeric_data[dataset_name][2], numeric_data[dataset_name][3], numeric_data[dataset_name][5] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_numeric_ncoref' + str(set_num)),'rb'))
        pad_sentences_training([string_data[dataset_name][0], string_data[dataset_name][2]], data_length['sequence_length_ment1'], vocabulary)
        pad_sentences_training([string_data[dataset_name][1], string_data[dataset_name][3]], data_length['sequence_length_ment2'], vocabulary)
        pad_sentences_training([string_data[dataset_name][4], string_data[dataset_name][6]], data_length['sequence_length_sents_ment1'], vocabulary)
        pad_sentences_training([string_data[dataset_name][5], string_data[dataset_name][7]], data_length['sequence_length_sents_ment2'], vocabulary)
        pad_sentences_training([string_data[dataset_name][8], string_data[dataset_name][10]], data_length['sequence_length_add_ment1'], vocabulary)
        pad_sentences_training([string_data[dataset_name][9], string_data[dataset_name][11]], data_length['sequence_length_add_ment2'], vocabulary)
        return [string_data, numeric_data, label_data]

def load_data_balance_ncoref(FLAGS, dataset_name, set_num, string_data, numeric_data, label_data, data_length, vocabulary):
    balace_dir = os.path.join(FLAGS.dataset_path, 'balance')
    if os.path.exists(balace_dir):
        print ("    restore balacned data of non-coref. on {} from {}, subset {}".format(dataset_name, FLAGS.dataset_path, set_num),)
        label_data[dataset_name][1] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_label_ncoref' + str(set_num)),'rb'))
        if FLAGS.used_mention == True:
            string_data[dataset_name][2], string_data[dataset_name][3] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_string_ncoref' + str(set_num)),'rb'))
        if FLAGS.used_sentence == True:
            string_data[dataset_name][6], string_data[dataset_name][7] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_string_sent_ncoref' + str(set_num)),'rb'))
        if FLAGS.used_addition == True:
            string_data[dataset_name][10], string_data[dataset_name][11] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_string_add_ncoref' + str(set_num)),'rb'))
        if FLAGS.used_numeric == True:
            numeric_data[dataset_name][2], numeric_data[dataset_name][3], numeric_data[dataset_name][5] = pk.load(open(os.path.join(balace_dir, dataset_name + '_data_numeric_ncoref' + str(set_num)),'rb'))
        pad_sentences_training([string_data[dataset_name][2]], data_length['sequence_length_ment1'], vocabulary)
        pad_sentences_training([string_data[dataset_name][3]], data_length['sequence_length_ment2'], vocabulary)
        pad_sentences_training([string_data[dataset_name][6]], data_length['sequence_length_sents_ment1'], vocabulary)
        pad_sentences_training([string_data[dataset_name][7]], data_length['sequence_length_sents_ment2'], vocabulary)
        pad_sentences_training([string_data[dataset_name][10]], data_length['sequence_length_add_ment1'], vocabulary)
        pad_sentences_training([string_data[dataset_name][11]], data_length['sequence_length_add_ment2'], vocabulary)
    return [string_data, numeric_data, label_data]

def split_value_process(x_ment):
    x_ment = [s.replace("\n", "").replace("\r","").strip() for s in x_ment]
    x_ment = [s.split(" ") for s in x_ment]
    x_ments = []
    for s in x_ment:
        x_s = []
        for w in s:
            x_s.append(float(w))
        x_ments.append(x_s)
    x_ment = np.array(x_ments)
    return x_ment

def load_numeric_data(FLAGS):
    if FLAGS.train_numeric_coref == None:
        x_train = None
        x_dev = None
        x_test = None
        return x_train, x_dev, x_test

    print ("  Load train dataset")
    train_coref_examples_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_numeric_coref + '_ment1')).readlines())
    train_coref_examples_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_numeric_coref + '_ment2')).readlines())
    train_coref_examples = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_numeric_coref)).readlines())
    train_ncoref_examples_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_numeric_ncoref + '_ment1')).readlines())
    train_ncoref_examples_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_numeric_ncoref + '_ment2')).readlines())
    train_ncoref_examples = list(open(os.path.join(FLAGS.dataset_path, FLAGS.train_numeric_ncoref)).readlines())

    print ("  Load development dataset")
    dev_coref_examples_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_numeric_coref + '_ment1')).readlines())
    dev_coref_examples_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_numeric_coref + '_ment2')).readlines())
    dev_coref_examples = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_numeric_coref)).readlines())
    dev_ncoref_examples_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_numeric_ncoref + '_ment1')).readlines())
    dev_ncoref_examples_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_numeric_ncoref + '_ment2')).readlines())
    dev_ncoref_examples = list(open(os.path.join(FLAGS.dataset_path, FLAGS.development_numeric_ncoref)).readlines())

    print ("  Load test dataset")
    test_coref_examples_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_numeric_coref + '_ment1')).readlines())
    test_coref_examples_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_numeric_coref + '_ment2')).readlines())
    test_coref_examples = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_numeric_coref)).readlines())
    test_ncoref_examples_ment1 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_numeric_ncoref + '_ment1')).readlines())
    test_ncoref_examples_ment2 = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_numeric_ncoref + '_ment2')).readlines())
    test_ncoref_examples = list(open(os.path.join(FLAGS.dataset_path, FLAGS.test_numeric_ncoref)).readlines())

    print ("  Split by value for train dataset")
    train_coref_examples_ment1 = split_value_process(train_coref_examples_ment1)
    train_coref_examples_ment2 = split_value_process(train_coref_examples_ment2)
    train_coref_examples = split_value_process(train_coref_examples)
    train_ncoref_examples_ment1 = split_value_process(train_ncoref_examples_ment1)
    train_ncoref_examples_ment2 = split_value_process(train_ncoref_examples_ment2)
    train_ncoref_examples = split_value_process(train_ncoref_examples)

    print ("  Split by value for development dataset")
    dev_coref_examples_ment1 = split_value_process(dev_coref_examples_ment1)
    dev_coref_examples_ment2 = split_value_process(dev_coref_examples_ment2)
    dev_coref_examples = split_value_process(dev_coref_examples)
    dev_ncoref_examples_ment1 = split_value_process(dev_ncoref_examples_ment1)
    dev_ncoref_examples_ment2 = split_value_process(dev_ncoref_examples_ment2)
    dev_ncoref_examples = split_value_process(dev_ncoref_examples)
    
    print ("  Split by value for test dataset")
    test_coref_examples_ment1 = split_value_process(test_coref_examples_ment1)
    test_coref_examples_ment2 = split_value_process(test_coref_examples_ment2)
    test_coref_examples = split_value_process(test_coref_examples)
    test_ncoref_examples_ment1 = split_value_process(test_ncoref_examples_ment1)
    test_ncoref_examples_ment2 = split_value_process(test_ncoref_examples_ment2)
    test_ncoref_examples = split_value_process(test_ncoref_examples)
    numeric_data = {}
    numeric_data['train'] = [train_coref_examples_ment1, train_coref_examples_ment2, train_ncoref_examples_ment1, train_ncoref_examples_ment2, train_coref_examples, train_ncoref_examples]
    numeric_data['dev'] = [dev_coref_examples_ment1, dev_coref_examples_ment2, dev_ncoref_examples_ment1, dev_ncoref_examples_ment2, dev_coref_examples, dev_ncoref_examples]
    numeric_data['test'] = [test_coref_examples_ment1, test_coref_examples_ment2, test_ncoref_examples_ment1, test_ncoref_examples_ment2, test_coref_examples, test_ncoref_examples]
    return numeric_data

def load_subset(FLAGS, string_data, numeric_data, label_data, dataset_name):
    x_ment1 = []
    x_ment2 = []
    x_sents_ment1 = []
    x_sents_ment2 = []
    x_add_ment1 = []
    x_add_ment2 = []
    x_numeric_ment1 = []
    x_numeric_ment2 = []
    x_numeric = []
    y = np.concatenate([label_data[dataset_name][0], label_data[dataset_name][1]], 0)
    
    if FLAGS.used_mention == True:
        x_ment1 = np.concatenate([string_data[dataset_name][0], string_data[dataset_name][2]], 0)
        x_ment2 = np.concatenate([string_data[dataset_name][1], string_data[dataset_name][3]], 0)
    if FLAGS.used_sentence == True:
        x_sents_ment1 = np.concatenate([string_data[dataset_name][4], string_data[dataset_name][6]], 0)
        x_sents_ment2 = np.concatenate([string_data[dataset_name][5], string_data[dataset_name][7]], 0)
    if FLAGS.used_addition == True:
        x_add_ment1 = np.concatenate([string_data[dataset_name][8], string_data[dataset_name][10]], 0)
        x_add_ment2 = np.concatenate([string_data[dataset_name][9], string_data[dataset_name][11]], 0)
    if FLAGS.used_numeric == True:
        x_numeric_ment1 = np.concatenate([numeric_data[dataset_name][0], numeric_data[dataset_name][2]], 0)
        x_numeric_ment2 = np.concatenate([numeric_data[dataset_name][1], numeric_data[dataset_name][3]], 0)
        x_numeric = np.concatenate([numeric_data[dataset_name][4], numeric_data[dataset_name][5]], 0)
    return y, x_ment1, x_ment2, x_sents_ment1, x_sents_ment2, x_add_ment1, x_add_ment2, x_numeric_ment1, x_numeric_ment2, x_numeric

def zip_features(x_train_ment1, x_train_ment2, x_train_sents_ment1, x_train_sents_ment2, x_train_add_ment1, x_train_add_ment2, x_train_numeric_ment1, x_train_numeric_ment2, x_train_numeric, y_train, FLAGS):
    used_data = []
    if FLAGS.used_mention == True:
        used_data.append(x_train_ment1)
        used_data.append(x_train_ment2)
    if FLAGS.used_sentence == True:
        used_data.append(x_train_sents_ment1)
        used_data.append(x_train_sents_ment2)
    if FLAGS.used_addition == True:
        used_data.append(x_train_add_ment1)
        used_data.append(x_train_add_ment2)
    if FLAGS.used_numeric == True:
        used_data.append(x_train_numeric_ment1)
        used_data.append(x_train_numeric_ment2)
        used_data.append(x_train_numeric)
    used_data.append(y_train)
    return list(zip(*used_data))
                
def unzip_features(batch, FLAGS):
    x_batch_ment1 = []
    x_batch_ment2 = []
    x_batch_sents_ment1 = []
    x_batch_sents_ment2 = []
    x_batch_add_ment1 = []
    x_batch_add_ment2 = []
    x_batch_numeric_ment1 = []
    x_batch_numeric_ment2 = []
    x_batch_numeric = []
    
    used_data = list(zip(*batch))
    index = 0
    if FLAGS.used_mention == True:
        x_batch_ment1 = used_data[index]
        index += 1
        x_batch_ment2 = used_data[index]
        index += 1
    if FLAGS.used_sentence == True:
        x_batch_sents_ment1 = used_data[index]
        index += 1
        x_batch_sents_ment2 = used_data[index]
        index += 1
    if FLAGS.used_addition == True:
        x_batch_add_ment1 = used_data[index]
        index += 1
        x_batch_add_ment2 = used_data[index]
        index += 1
    if FLAGS.used_numeric == True:
        x_batch_numeric_ment1 = used_data[index]
        index += 1
        x_batch_numeric_ment2 = used_data[index]
        index += 1
        x_batch_numeric = used_data[index]
        index += 1
    y_batch = used_data[-1]
    return x_batch_ment1, x_batch_ment2, x_batch_sents_ment1, x_batch_sents_ment2, x_batch_add_ment1, x_batch_add_ment2, x_batch_numeric_ment1, x_batch_numeric_ment2, x_batch_numeric, y_batch

def batch_iter_cnn_fnn(data, batch_size, ):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) if len(data)%batch_size == 0 else (int(len(data)/batch_size)+1) 
    print ("Total batches {}".format(num_batches_per_epoch + 1))
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_data = data[shuffle_indices]
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        print("batch {} ".format(batch_num + 1)),
        yield shuffled_data[start_index:end_index]

def data_smapling(data, num_smaple, rand=True):
    sample_list = []
    data_size = len(data)
    if rand == True:
        print ("  Random shuffled data")
        shuffle_indices = np.random.permutation(np.arange(data_size))
    elif rand == False:
        shuffle_indices = range(data_size)
    num_sets = int(len(data)/num_smaple) if (len(data) % num_smaple) == 0 else int(len(data)/num_smaple)+ 1
    for set_num in range(num_sets):
        start_index = set_num * num_smaple
        end_index = min((set_num + 1) * num_smaple, data_size)
        sample_list.append(shuffle_indices[start_index:end_index])
    return sample_list, num_sets

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def save_balacne_data(mention_pair_key_data, string_data, numeric_data, label_data, vocabulary, vocabulary_inv, data_length, dataset_path):
    balace_dir = os.path.join(dataset_path, 'balance')
    print ("Saving at ", balace_dir)
    if not os.path.exists(balace_dir):
        os.makedirs(balace_dir)
    pk.dump((vocabulary, vocabulary_inv),open(os.path.join(balace_dir, 'vocabulary'),'wb'))
    pk.dump(data_length,open(os.path.join(balace_dir, 'data_length'),'wb'))
    train_balance_counts = min(len(label_data['train'][0]), len(label_data['train'][0]))
    dev_balance_counts = min(len(label_data['dev'][0]), len(label_data['dev'][0]))
    test_balance_counts = min(len(label_data['test'][0]), len(label_data['test'][0]))
    print ("  on train dataset")
    data_balance_sets = {}
    if len(label_data['train'][0]) == train_balance_counts:
        pk.dump((label_data['train'][0]),open(os.path.join(balace_dir, 'train_data_label_coref'),'wb'))
        pk.dump((string_data['train'][0], string_data['train'][1]),open(os.path.join(balace_dir, 'train_data_string_coref'),'wb'))
        pk.dump((string_data['train'][4], string_data['train'][5]),open(os.path.join(balace_dir, 'train_data_string_sent_coref'),'wb'))
        pk.dump((string_data['train'][8], string_data['train'][9]),open(os.path.join(balace_dir, 'train_data_string_add_coref'),'wb'))      
        pk.dump((numeric_data['train'][0], numeric_data['train'][1],  numeric_data['train'][4]),open(os.path.join(balace_dir, 'train_data_numeric_coref'),'wb'))
        pk.dump((mention_pair_key_data['train'][0]),open(os.path.join(balace_dir, 'train_data_mention_pair_key_coref'),'wb'))
        print (type(numeric_data['train'][0]))
        lengths = len(label_data['train'][1])
        shuffle_indices = np.random.permutation(np.arange(lengths))        
        num_sets = int(lengths/train_balance_counts) if lengths%train_balance_counts == 0 else (int(lengths/train_balance_counts)+1)
        data_balance_sets['train'] = num_sets
        sample_list = []
        for set_num in range(num_sets):
            start_index = set_num * train_balance_counts
            end_index = min((set_num + 1) * train_balance_counts, lengths)
            sample_list.append(shuffle_indices[start_index:end_index])
        for set_num in range(num_sets):
            pk.dump((label_data['train'][1][sample_list[set_num]]),open(os.path.join(balace_dir, 'train_data_label_ncoref' + str(set_num)),'wb'))
            pk.dump((string_data['train'][2][sample_list[set_num]], string_data['train'][3][sample_list[set_num]]),open(os.path.join(balace_dir, 'train_data_string_ncoref' + str(set_num)),'wb'))
            pk.dump((string_data['train'][6][sample_list[set_num]], string_data['train'][7][sample_list[set_num]]),open(os.path.join(balace_dir, 'train_data_string_sent_ncoref' + str(set_num)),'wb'))
            pk.dump((string_data['train'][10][sample_list[set_num]], string_data['train'][11][sample_list[set_num]]),open(os.path.join(balace_dir, 'train_data_string_add_ncoref' + str(set_num)),'wb'))
            pk.dump((numeric_data['train'][2][sample_list[set_num]], numeric_data['train'][3][sample_list[set_num]], numeric_data['train'][5][sample_list[set_num]]),open(os.path.join(balace_dir, 'train_data_numeric_ncoref' + str(set_num)),'wb'))
            pk.dump((mention_pair_key_data['train'][1][sample_list[set_num]]),open(os.path.join(balace_dir, 'train_data_mention_pair_key_ncoref' + str(set_num)),'wb'))

    print ("  on dev dataset")
    if len(label_data['dev'][0]) == dev_balance_counts:
        pk.dump((label_data['dev'][0]),open(os.path.join(balace_dir, 'dev_data_label_coref'),'wb'))
        pk.dump((string_data['dev'][0], string_data['dev'][1]),open(os.path.join(balace_dir, 'dev_data_string_coref'),'wb'))
        pk.dump((string_data['dev'][4], string_data['dev'][5]),open(os.path.join(balace_dir, 'dev_data_string_sent_coref'),'wb'))
        pk.dump((string_data['dev'][8], string_data['dev'][9]),open(os.path.join(balace_dir, 'dev_data_string_add_coref'),'wb'))   
        pk.dump((numeric_data['dev'][0], numeric_data['dev'][1], numeric_data['dev'][4]),open(os.path.join(balace_dir, 'dev_data_numeric_coref'),'wb'))
        pk.dump((mention_pair_key_data['dev'][0]),open(os.path.join(balace_dir, 'dev_data_mention_pair_key_coref'),'wb'))
        print (type(numeric_data['dev'][0]))
        lengths = len(label_data['dev'][1])
        shuffle_indices = np.random.permutation(np.arange(lengths))
        num_sets = int(lengths/dev_balance_counts) if lengths%dev_balance_counts == 0 else (int(lengths/dev_balance_counts)+1)
        data_balance_sets['dev'] = num_sets
        sample_list = []
        for set_num in range(num_sets):
            start_index = set_num * dev_balance_counts
            end_index = min((set_num + 1) * dev_balance_counts, lengths)
            sample_list.append(shuffle_indices[start_index:end_index])
        for set_num in range(num_sets):
            pk.dump((label_data['dev'][1][sample_list[set_num]]),open(os.path.join(balace_dir, 'dev_data_label_ncoref' + str(set_num)),'wb'))
            pk.dump((string_data['dev'][2][sample_list[set_num]], string_data['dev'][3][sample_list[set_num]]),open(os.path.join(balace_dir, 'dev_data_string_ncoref' + str(set_num)),'wb'))
            pk.dump((string_data['dev'][6][sample_list[set_num]], string_data['dev'][7][sample_list[set_num]]),open(os.path.join(balace_dir, 'dev_data_string_sent_ncoref' + str(set_num)),'wb'))
            pk.dump((string_data['dev'][10][sample_list[set_num]], string_data['dev'][11][sample_list[set_num]]),open(os.path.join(balace_dir, 'dev_data_string_add_ncoref' + str(set_num)),'wb'))
            pk.dump((numeric_data['dev'][2][sample_list[set_num]], numeric_data['dev'][3][sample_list[set_num]], numeric_data['dev'][5][sample_list[set_num]]),open(os.path.join(balace_dir, 'dev_data_numeric_ncoref' + str(set_num)),'wb'))
            
            pk.dump((mention_pair_key_data['dev'][1][sample_list[set_num]]),open(os.path.join(balace_dir, 'dev_data_mention_pair_key_ncoref' + str(set_num)),'wb'))

    print ("  on test dataset")
    if len(label_data['test'][0]) == test_balance_counts:
        pk.dump((label_data['test'][0]),open(os.path.join(balace_dir, 'test_data_label_coref'),'wb'))
        pk.dump((string_data['test'][0], string_data['test'][1]),open(os.path.join(balace_dir, 'test_data_string_coref'),'wb'))
        pk.dump((string_data['test'][4], string_data['test'][5]),open(os.path.join(balace_dir, 'test_data_string_sent_coref'),'wb'))
        pk.dump((string_data['test'][8], string_data['test'][9]),open(os.path.join(balace_dir, 'test_data_string_add_coref'),'wb'))
        pk.dump((numeric_data['test'][0], numeric_data['test'][1], numeric_data['test'][4]),open(os.path.join(balace_dir, 'test_data_numeric_coref'),'wb'))
        pk.dump((mention_pair_key_data['test'][0]),open(os.path.join(balace_dir, 'test_data_mention_pair_key_coref'),'wb'))
        print (type(numeric_data['test'][0]))
        lengths = len(label_data['test'][1])
        shuffle_indices = np.random.permutation(np.arange(lengths))   
        num_sets = int(lengths/test_balance_counts) if lengths%test_balance_counts == 0 else (int(lengths/test_balance_counts)+1)
        data_balance_sets['test'] = num_sets
        sample_list = []
        for set_num in range(num_sets):
            start_index = set_num * test_balance_counts
            end_index = min((set_num + 1) * test_balance_counts, lengths)
            sample_list.append(shuffle_indices[start_index:end_index])
        for set_num in range(num_sets):
            pk.dump((label_data['test'][1][sample_list[set_num]]),open(os.path.join(balace_dir, 'test_data_label_ncoref' + str(set_num)),'wb'))
            pk.dump((string_data['test'][2][sample_list[set_num]], string_data['test'][3][sample_list[set_num]]),open(os.path.join(balace_dir, 'test_data_string_ncoref' + str(set_num)),'wb'))
            pk.dump((string_data['test'][6][sample_list[set_num]], string_data['test'][7][sample_list[set_num]]),open(os.path.join(balace_dir, 'test_data_string_sent_ncoref' + str(set_num)),'wb'))
            pk.dump((string_data['test'][10][sample_list[set_num]], string_data['test'][11][sample_list[set_num]]),open(os.path.join(balace_dir, 'test_data_string_add_ncoref' + str(set_num)),'wb'))
            pk.dump((numeric_data['test'][2][sample_list[set_num]], numeric_data['test'][3][sample_list[set_num]], numeric_data['test'][5][sample_list[set_num]]),open(os.path.join(balace_dir, 'test_data_numeric_ncoref' + str(set_num)),'wb'))
            pk.dump((mention_pair_key_data['test'][1][sample_list[set_num]]),open(os.path.join(balace_dir, 'test_data_mention_pair_key_ncoref' + str(set_num)),'wb'))

    pk.dump(data_balance_sets,open(os.path.join(balace_dir, 'data_balance_sets'),'wb'))
    print ("each subset on train/dev/test has ", train_balance_counts, dev_balance_counts, test_balance_counts)
    print ("each dataset on train/dev/test has ", data_balance_sets)
    print ("the balance data has saved at ", balace_dir)

if __name__ == "__main__":
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

    FLAGS, unparsed = parser.parse_known_args()
    print (FLAGS.dataset_path)
    load_data(FLAGS)

