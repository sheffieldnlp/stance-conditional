__author__ = 'Isabelle Augenstein'

import tensorflow as tf
import numpy as np

from naga.shared.trainer import Trainer
from tfrnn.rnn import Encoder
from tfrnn.rnn import Projector
from tfrnn.rnn import rnn_cell
from tfrnn.hooks import SaveModelHook
from tfrnn.hooks import AccuracyHook
from tfrnn.hooks import LossHook
from tfrnn.hooks import SpeedHook
from tfrnn.batcher import BatchBucketSampler
from tfrnn.util import sample_one_hot
from tfrnn.util import debug_node
from tfrnn.hooks import LoadModelHook

from readwrite import reader
from preprocess import tokenise_tweets, build_dataset, transform_tweet, transform_labels

def get_model(batch_size, max_seq_length, input_size, hidden_size, target_size,
              vocab_size):
    # batch_size x max_seq_length
    inputs = tf.placeholder(tf.int32, [batch_size, max_seq_length])
    inputs_cond = tf.placeholder(tf.int32, [batch_size, max_seq_length])

    embedding_matrix = tf.Variable(tf.random_normal([vocab_size, input_size]),  #input_size is embeddings size
                                   name="embedding_matrix")

    # batch_size x max_seq_length x input_size
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)
    embedded_inputs_cond = tf.nn.embedding_lookup(embedding_matrix, inputs_cond)

    # print_inputs_shapes = tf.Print(embedded_inputs, [inputs.get_shape(),
    #                                                  embedded_inputs.get_shape()],
    #                                "input to embedding shape: ")
    #
    # debug_node(print_inputs_shapes, feed_dict={
    #     inputs: np.random.randint(vocab_size, size=(batch_size, max_seq_length)),
    #     inputs_cond: np.random.randint(vocab_size, size=(batch_size, max_seq_length))
    # })

    # [batch_size x inputs_size] with max_seq_length elements
    # fixme: possibly inefficient
    # inputs_list[0]: batch_size x input[0] <-- word vector of the first word
    inputs_list = [tf.squeeze(x) for x in
                   tf.split(1, max_seq_length, embedded_inputs)]
    inputs_cond_list = [tf.squeeze(x) for x in
                        tf.split(1, max_seq_length, embedded_inputs_cond)]

    # print_inputs_list = tf.Print(inputs_list[0], [x.get_shape() for x in inputs_list], "inputs list")
    #
    # debug_node(print_inputs_list, feed_dict={
    #     inputs: np.random.randint(vocab_size, size=(batch_size, max_seq_length)),
    #     inputs_cond: np.random.randint(vocab_size, size=(batch_size, max_seq_length))
    # })

    lstm_encoder = Encoder(rnn_cell.BasicLSTMCell, input_size, hidden_size)
    start_state = tf.zeros([batch_size, lstm_encoder.state_size])

    # [h_i], [h_i, c_i] <-- LSTM
    # [h_i], [h_i] <-- RNN
    outputs, states = lstm_encoder(inputs_list, start_state, "LSTM")

    # running a second LSTM conditioned on the last state of the first
    outputs_cond, states_cond = lstm_encoder(inputs_cond_list, states[-1],
                                             "LSTMcond")

    model = Projector(target_size, non_linearity=tf.nn.tanh)(outputs_cond[-1])

    return model, [inputs, inputs_cond]


def test_trainer(tweets, targets, labels, dictionary):
    # parameters
    num_samples = 2113
    max_epochs = 100
    learning_rate = 0.01
    batch_size = 129  # number training examples per training epoch
    input_size = 91
    hidden_size = 83


    # synthetic data example from Tim below
    #target_size = 3  # number of different labels
    #vocab_size = 13
    #max_seq_length = 17  # max word length of sentence. Divide this into seq length for tweet and target later.
    #data = [
    #    np.random.choice(vocab_size, [num_samples, max_seq_length]),  # create 2113 samples of length 17 with indeces between 0 in 12
    #    np.random.choice(vocab_size, [num_samples, max_seq_length]),  # create 2113 samples of length 17 with indeces between 0 in 12
    #    np.asarray([sample_one_hot(target_size) for i in range(0, num_samples)])   # one hot vector for labels
    #]

    # real data example from Tim below
        # directory = "../data/rte_engineering/"
        # data, vocab = load_engineering_rte_data(directory)
        # print(data[0])
        # vocab_size = vocab.size()
        # target_size = len(data[0][2])
        # max_seq_length = max(len(data[0][0]), len(data[0][1]))

    # real data stance-semeval
    target_size = 3
    max_seq_length = len(tweets[0])
    vocab_size = dictionary.__sizeof__()
    data = [tweets, targets, labels]


    # output of get_model(): model, [inputs, inputs_cond]
    model, placeholders = get_model(batch_size, max_seq_length, input_size,
                                    hidden_size, target_size, vocab_size)

    targets = tf.placeholder(tf.float32, [batch_size, target_size], "targets")
    loss = tf.nn.softmax_cross_entropy_with_logits(model, targets)   # targets: labels (e.g. pos/neg/neutral)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    batcher = BatchBucketSampler(data, batch_size)
    acc_batcher = BatchBucketSampler(data, batch_size)

    placeholders += [targets]

    hooks = [
        SpeedHook(50, batch_size),
        SaveModelHook("./out/save", 100),
        #LoadModelHook("./out/save", 10),
        AccuracyHook(acc_batcher, placeholders, 5),
        LossHook(10)
    ]

    trainer = Trainer(optimizer, max_epochs, hooks)

    trainer(batcher, placeholders=placeholders, loss=loss, model=model)


if __name__ == '__main__':

    tweets, targets, labels = reader.readTweetsOfficial("../data/semeval2016-task6-train+dev.txt")
    tweet_tokens = tokenise_tweets(tweets)
    target_tokens = tokenise_tweets(targets)
    count, dictionary, reverse_dictionary = build_dataset([token for senttoks in tweet_tokens+target_tokens for token in senttoks])  #flatten tweets for vocab construction
    transformed_tweets = [transform_tweet(dictionary, senttoks) for senttoks in tweet_tokens]
    transformed_targets = [transform_tweet(dictionary, senttoks) for senttoks in target_tokens]
    transformed_labels = transform_labels(labels)

    test_trainer(transformed_tweets, transformed_targets, transformed_labels, dictionary)
