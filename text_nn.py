# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

__author__ = 'Jheng-Long Wu'

import numpy as np
import math
import tensorflow as tf
import tensorflow.python.platform

class TextCNNFNN(object):
    """
    A CNN with FNN for coreference resolution
    Uses an embedding layer, followed by a 
    convolutional, max-pooling, fully connected and softmax layer.
    """
    def __init__(
      self, sequence_length_ment1, sequence_length_ment2, 
      sequence_length_sents_ment1, sequence_length_sents_ment2, 
      sequence_length_add_ment1, sequence_length_add_ment2,
      sequence_length_numeric_ment1, sequence_length_numeric_ment2, 
      sequence_length_numeric, num_classes, pretrained_embedding, 
      embedding_size, vocab_size, filter_sizes, 
      num_filters, num_dim_hidden, num_cnn_layers, num_fnn_layers, 
      used_mention, used_sentence, used_addition, used_numeric, 
      l2_reg_lambda):
        print ("\n--------------Text%sCNN%sFNN----------------" % (num_cnn_layers,num_fnn_layers))
        # Placeholders for input, output and dropout
        self.input_ment1_x = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length_ment1], name="String_mention1")
        self.input_ment2_x = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length_ment2], name="String_mention2")
        self.input_sents_ment1_x = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length_sents_ment1], name="String_sentence_mention1")
        self.input_sents_ment2_x = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length_sents_ment2], name="String_sentence_mention2")
        self.input_add_ment1_x = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length_add_ment1], name="String_addition_mention1")
        self.input_add_ment2_x = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length_add_ment2], name="String_addition_mention2")
        self.input_numeric_ment1_x = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length_numeric_ment1], name="Numeric_mention1")
        self.input_numeric_ment2_x = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length_numeric_ment2], name="Numeric_mention2")
        self.input_numeric_x = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length_numeric], name="Numeric")
        self.input_y = tf.placeholder(dtype=tf.float32, shape=None, name="Label_class")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        # Embedding layer
        if used_mention == True or used_sentence == True or used_addition == True:
            with tf.device('/cpu:0'), tf.name_scope("word-embedding"):
                if pretrained_embedding == None:
                    emb = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=True, name="emb")
                # Use pre-trained word embedding
                if pretrained_embedding != None:
                    emb = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=False, name="emb")
                    emb.assign(pretrained_embedding)

        # Convolution layer
        conv_pooled_outputs = []
        # Used mention inforamtion
        if used_mention == True:
            # Mention 1: Looking up form embedding for string of mention
            self.embedded_chars_ment1 = tf.nn.embedding_lookup(emb, self.input_ment1_x)
            print (self.embedded_chars_ment1)
            self.embedded_chars_expanded_ment1 = tf.expand_dims(self.embedded_chars_ment1, -1)
            print (self.embedded_chars_expanded_ment1)
            # Mention 1: Create a convolution + maxpool layer for each filter size
            for i, filter_size in enumerate(filter_sizes):
                if filter_size > sequence_length_ment1:
                   print ("the filter size {} need less than {}".format(sequence_length_ment1)) 
                   exit()
                with tf.device('/gpu:0'), tf.name_scope("M1-conv1-filter%s"%(filter_size)):
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    conv_pooled_ment1 = self._Build_CNN_MAXPOOL_Model(self.embedded_chars_expanded_ment1,filter_shape)
                    print(conv_pooled_ment1.op.name, ' ', conv_pooled_ment1.get_shape().as_list())
                for layer in range(1, num_cnn_layers):
                    #convolution + maxpool layers by auto loop
                    with tf.device('/gpu:0'), tf.name_scope("M1-conv%s-filter%s" % (layer+1,filter_size)):
                        d = conv_pooled_ment1.get_shape()[1:].as_list()[0]
                        filter_shape = [min(filter_size*2, d), 1, num_filters, num_filters]
                        if layer+1 != num_cnn_layers:
                            conv_pooled_ment1 = self._Build_CNN_MAXPOOL_Model(conv_pooled_ment1,filter_shape)
                        else:
                            conv_pooled_ment1 = self._Build_CNN_MAXPOOL_Model(conv_pooled_ment1,filter_shape, ksize_all=True)
                        print(conv_pooled_ment1.op.name, ' ', conv_pooled_ment1.get_shape().as_list())
                conv_pooled_outputs.append(self._reshape_output_cnn(conv_pooled_ment1))

            # Mention 2: Looking up form embedding for string of mention
            self.embedded_chars_ment2 = tf.nn.embedding_lookup(emb, self.input_ment2_x)
            print (self.embedded_chars_ment2)
            self.embedded_chars_expanded_ment2 = tf.expand_dims(self.embedded_chars_ment2, -1)
            print (self.embedded_chars_expanded_ment2)
            # Mention 2: Create a convolution + maxpool layer for each filter size
            for i, filter_size in enumerate(filter_sizes):
                if filter_size > sequence_length_ment2:
                   print ("the filter size {} need less than {}".format(sequence_length_ment2))
                   exit()
                with tf.device('/gpu:1'), tf.name_scope("M2-conv1-filter%s"%(filter_size)):
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    conv_pooled_ment2 = self._Build_CNN_MAXPOOL_Model(self.embedded_chars_expanded_ment2,filter_shape)
                    print(conv_pooled_ment2.op.name, ' ', conv_pooled_ment2.get_shape().as_list())
                for layer in range(1, num_cnn_layers):
                    #convolution + maxpool layers by auto loop
                    with tf.device('/gpu:1'), tf.name_scope("M2-conv%s-fliter%s" % (layer+1, filter_size)):
                        d = conv_pooled_ment2.get_shape()[1:].as_list()[0]
                        filter_shape = [min(filter_size*2, d), 1, num_filters, num_filters]
                        if layer+1 != num_cnn_layers:
                            conv_pooled_ment2 = self._Build_CNN_MAXPOOL_Model(conv_pooled_ment2,filter_shape)
                        else:
                            conv_pooled_ment2 = self._Build_CNN_MAXPOOL_Model(conv_pooled_ment2,filter_shape, ksize_all=True)
                        print(conv_pooled_ment2.op.name, ' ', conv_pooled_ment2.get_shape().as_list())
                conv_pooled_outputs.append(self._reshape_output_cnn(conv_pooled_ment2))

        # Used mention sentences information
        if used_sentence == True:
            # Looking up form embedding for string of sentence of mention
            self.embedded_chars_sents_ment1 = tf.nn.embedding_lookup(emb, self.input_sents_ment1_x)
            self.embedded_chars_expanded_sents_ment1 = tf.expand_dims(self.embedded_chars_sents_ment1, -1)
            self.embedded_chars_sents_ment2 = tf.nn.embedding_lookup(emb, self.input_sents_ment2_x)
            self.embedded_chars_expanded_sents_ment2 = tf.expand_dims(self.embedded_chars_sents_ment2, -1)
            conv_collection = []
            # Metion 1 sentence: Create a convolution + maxpool layer for each filter size
            for i, filter_size in enumerate(filter_sizes):
                if filter_size > sequence_length_sents_ment1:
                   print ("the filter size {} need less than {}".format(sequence_length_sents_ment1))
                   exit()
                with tf.name_scope("M1-conv1-sent-%s" % (i + 1)):
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    conv_pooled_sents_ment1 = self._Build_CNN_MAXPOOL_Model(self.embedded_chars_expanded_sents_ment1,filter_shape)
                    print(conv_pooled_sents_ment1.op.name, ' ', conv_pooled_sents_ment1.get_shape().as_list())
                #convolution + maxpool layers by auto loop
                for layer in range(1, num_cnn_layers):
                    with tf.name_scope("M1-conv%s-sent-%s" % (layer+1, i + 1)):
                        d = conv_pooled_sents_ment1.get_shape()[1:].as_list()[0]
                        filter_shape = [min(filter_size*2, d), 1, num_filters, num_filters]
                        if layer+1 != num_cnn_layers:
                            conv_pooled_sents_ment1 = self._Build_CNN_MAXPOOL_Model(conv_pooled_sents_ment1,filter_shape)
                        else:
                            conv_pooled_sents_ment1 = self._Build_CNN_MAXPOOL_Model(conv_pooled_sents_ment1,filter_shape, ksize_all=True)
                        print(conv_pooled_sents_ment1.op.name, ' ', conv_pooled_sents_ment1.get_shape().as_list())
                    
                conv_pooled_outputs.append(self._reshape_output_cnn(conv_pooled_sents_ment1))

            # Metion 2 sentence: Create a convolution + maxpool layer for each filter size
            for i, filter_size in enumerate(filter_sizes):
                if filter_size > sequence_length_sents_ment2:
                   print ("the filter size {} need less than {}".format(sequence_length_sents_ment2))
                   exit()
                with tf.name_scope("M2-conv1-sent-%s" % (i + 1)):
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    conv_pooled_sents_ment2 = self._Build_CNN_MAXPOOL_Model(self.embedded_chars_expanded_sents_ment2,filter_shape)
                    print(conv_pooled_sents_ment2.op.name, ' ', conv_pooled_sents_ment2.get_shape().as_list())
                #convolution + maxpool layers by auto loop
                for layer in range(1, num_cnn_layers):
                    with tf.name_scope("M2-conv%s-sent-%s" % (layer+1, i + 1)):
                        d = conv_pooled_sents_ment2.get_shape()[1:].as_list()[0]
                        filter_shape = [min(filter_size*2, d), 1, num_filters, num_filters]
                        if layer+1 != num_cnn_layers:
                            conv_pooled_sents_ment2 = self._Build_CNN_MAXPOOL_Model(conv_pooled_sents_ment2,filter_shape)
                        else:
                            conv_pooled_sents_ment2 = self._Build_CNN_MAXPOOL_Model(conv_pooled_sents_ment2,filter_shape, ksize_all=True)
                        print(conv_pooled_sents_ment2.op.name, ' ', conv_pooled_sents_ment2.get_shape().as_list())
                conv_pooled_outputs.append(self._reshape_output_cnn(conv_pooled_sents_ment2))

        # used mention addition information
        if used_addition == True:
            self.embedded_chars_add_ment1 = tf.nn.embedding_lookup(emb, self.input_add_ment1_x, name="addition_embedding_ment1")
            print (self.embedded_chars_add_ment1)
            self.embedded_chars_add_ment2 = tf.nn.embedding_lookup(emb, self.input_add_ment2_x, name="addition_embedding_ment2")
            print (self.embedded_chars_add_ment2)
            self.conv_pooled_add = tf.concat([self._reshape_output_cnn(self.embedded_chars_add_ment1), self._reshape_output_cnn(self.embedded_chars_add_ment2)], 1, name="concat_addition_features")
            print (self.conv_pooled_add)
            # Hiddend layers by auto loop
            for layer in range(int(num_fnn_layers/2)):
                with tf.variable_scope("FNN_RELU_addition%s" % (layer +1)) as scope:
                    dim_add_features = self.conv_pooled_add.get_shape().as_list()[-1]
                    shape_h =[dim_add_features, max(int(dim_add_features/2), num_dim_hidden)]
                    self.conv_pooled_add = self._Build_FNN_Model(self.conv_pooled_add, 
                        weight_shape=shape_h)
                    print(self.conv_pooled_add.op.name, ' ', self.conv_pooled_add.get_shape().as_list())
            conv_pooled_outputs.append(self.conv_pooled_add)
 
        # Fully connected layer for string of mention.
        if used_mention == True or used_addition == True:
            for i in range(len(conv_pooled_outputs)):
                print (conv_pooled_outputs[i])
            self.string_discoures_features = tf.concat(conv_pooled_outputs, 1, name="concat_mentions")
            print(self.string_discoures_features.op.name, ' ', self.string_discoures_features.get_shape().as_list())
            # Hiddend layers by auto loop
            for layer in range(int(num_fnn_layers/2)):
                with tf.variable_scope("FNN_RELU_string%s" % (layer +1)) as scope:
                    dim_mention_features = self.string_discoures_features.get_shape().as_list()[-1]
                    shape_h =[dim_mention_features, max(int(dim_mention_features/2), num_dim_hidden)]
                    self.string_discoures_features = self._Build_FNN_Model(self.string_discoures_features, 
                        weight_shape=shape_h)
                    print(self.string_discoures_features.op.name, ' ', self.string_discoures_features.get_shape().as_list())

        # Fully connected layer for numeric of mention.
        if used_numeric == True:
            #numeric_features = [self.input_numeric_ment1_x, self.input_numeric_ment2_x]
            numeric_features = [self.input_numeric_ment1_x, self.input_numeric_ment2_x, self.input_numeric_x]
            for i in range(len(numeric_features)):
                print (numeric_features[i])
            self.FNN_RELU_numeric = tf.concat(numeric_features, 1, name="concat_numeric_features")
            print(self.FNN_RELU_numeric.op.name, ' ', self.FNN_RELU_numeric.get_shape().as_list())

            # Fully connected layer.
            for layer in range(int(num_fnn_layers/2)):
                with tf.variable_scope("FNN_RELU_numeric%s" % (layer +1)) as scope:
                    dim_numeric_features = self.FNN_RELU_numeric.get_shape().as_list()[-1]
                    shape_h = [dim_numeric_features, max(int(dim_numeric_features/2), num_dim_hidden)]
                    self.FNN_RELU_numeric = self._Build_FNN_Model(self.FNN_RELU_numeric, 
                        weight_shape=shape_h)
                    print(self.FNN_RELU_numeric.op.name, ' ', self.FNN_RELU_numeric.get_shape().as_list())

        # Union two fnns of string and numeric
        with tf.name_scope("union"):
            union_inputs_fnn = []
            if used_mention == True or used_sentence == True:
                union_inputs_fnn = [self.string_discoures_features]
            if used_numeric == True: 
                union_inputs_fnn.append(self.FNN_RELU_numeric)
            for i in range(len(union_inputs_fnn)):
                print(union_inputs_fnn[i], ' ', union_inputs_fnn[i].get_shape().as_list())
            self.FNN_RELU = tf.concat(union_inputs_fnn, 1, name='union')
            print(self.FNN_RELU.op.name, ' ', self.FNN_RELU.get_shape().as_list())

        # Fully connected layer.
        for layer in range(num_fnn_layers):
            with tf.variable_scope("FNN_RELU%s" % (layer +1)) as scope:
                dim_union = self.FNN_RELU.get_shape().as_list()[-1]
                shape_h = [dim_union, max(int(dim_union/2), num_dim_hidden)]
                self.FNN_RELU = self._Build_FNN_Model(self.FNN_RELU, 
                    weight_shape=shape_h)
                print(self.FNN_RELU.op.name, ' ', self.FNN_RELU.get_shape().as_list())

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.FNN_RELU, self.dropout_keep_prob)
            print (self.h_drop)
        
        # Output layer that softmax, i.e. softmax(WX + b)0l
        with tf.variable_scope('output') as scope:
            self.weights_output = tf.Variable(tf.truncated_normal([num_dim_hidden, num_classes], stddev=0.1), name="weights")
            self.biases_output = self._variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))
            self.logits = tf.add(tf.matmul(self.h_drop, self.weights_output), self.biases_output, name="scores")
            print (self.logits)

        # Calculate the average cross entropy loss across the batch.
        with tf.name_scope("loss"):
            l2_loss = tf.constant(0.0) # Keeping track of l2 regularization loss (optional)
            l2_loss += tf.nn.l2_loss(self.weights_output)
            l2_loss += tf.nn.l2_loss(self.biases_output)
            #self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y, name="softmax")
            self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y, name="softmax")
            self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss
            print (self.loss)
        
        # Score
        with tf.name_scope("score"):
            self.scores = tf.nn.softmax(self.logits)
            print (self.scores)
        
        # Prediction
        with tf.name_scope("prediction"):
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            print (self.predictions)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1, name="gold"))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            print (self.accuracy)
        print ("\n--------------Text%sCNN%sFNN----------------" % (num_cnn_layers,num_fnn_layers))
        return 

    def _Build_CNN_MAXPOOL_Model(self, conv4D, filter_shape, ksize_all=False):
        # Weight
        W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1), name="W")
        # Biase
        b = tf.Variable(tf.constant(0.1, shape=[filter_shape[-1]]), name="b")
        # Build convolution NN
        conv = tf.nn.conv2d(
                            conv4D,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv")
        # Apply nonlinearity
        conv_relu = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # All pool or not
        if ksize_all:
            d = conv_relu.get_shape()[1:].as_list()[0]
        else:
            d = conv_relu.get_shape()[1:].as_list()[0]/2
        d = 1 if d < 1 else d # avoid d less than zero
        # Maxpooling over the CNN outputs
        conv_pooled = tf.nn.max_pool(
                            conv_relu,
                            ksize=[1, d, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool")
        return conv_pooled
    
    def _Build_FNN_Model(self, input_layer, weight_shape):
        weights = tf.Variable(
            tf.truncated_normal(shape=weight_shape, 
                                stddev=1.0 / math.sqrt(float(weight_shape[-1]))))
        biases = self._variable_on_cpu('biases', [weight_shape[-1]], tf.constant_initializer(0.1))
        return tf.nn.relu(tf.matmul(input_layer, weights) + biases)

    def _variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _reshape_output_cnn(self, obj_cnn):
        dim = obj_cnn.get_shape().as_list()
        if len(dim) == 4:
            return tf.reshape(obj_cnn, [-1, dim[1]*dim[2]*dim[3]])
        if len(dim) == 3:
            return tf.reshape(obj_cnn, [-1, dim[1]*dim[2]])

if __name__ == '__main__':
    # test to crate TextCNNFNN object
    cnn = TextCNNFNN(sequence_length_ment1=7,
                    sequence_length_ment2=8,
                    sequence_length_sents_ment1=15,
                    sequence_length_sents_ment2=17,
                    sequence_length_add_ment1=1,
                    sequence_length_add_ment2=1,
                    sequence_length_numeric_ment1=3,
                    sequence_length_numeric_ment2=3,
                    sequence_length_numeric=2,
                    num_classes=2,
                    pretrained_embedding=None,
                    embedding_size=300,
                    vocab_size=5000,
                    filter_sizes=[2],
                    num_filters=200,
                    num_dim_hidden=200,
                    num_cnn_layers=5,
                    num_fnn_layers=10,
                    used_mention=True,
                    used_sentence=True,
                    used_addition=True,
                    used_numeric=True,
                    l2_reg_lambda=0)