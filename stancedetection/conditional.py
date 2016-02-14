import tensorflow as tf
import numpy as np

from naga.shared.trainer import Trainer
from tfrnn.rnn import Encoder, Projector, rnn_cell
from tfrnn.hooks import SaveModelHook, AccuracyHook, LossHook, SpeedHook
from tfrnn.batcher import BatchBucketSampler
from tfrnn.util import sample_one_hot, debug_node, load_model
from tfrnn.hooks import LoadModelHook
from readwrite import reader, writer
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


def test_trainer(dictionary, tweets, targets, labels, ids, tweets_test, targets_test, labels_test, ids_test):
    # parameters
    num_samples = 2113
    max_epochs = 6  # 100
    learning_rate = 0.01
    batch_size = 97  # number training examples per training epoch
    input_size = 91
    hidden_size = 83


    # real data stance-semeval
    target_size = 3
    max_seq_length = len(tweets[0])
    vocab_size = len(dictionary)
    data = [np.asarray(tweets), np.asarray(targets), np.asarray(ids), np.asarray(labels)]


    # output of get_model(): model, [inputs, inputs_cond]
    model, placeholders = get_model(batch_size, max_seq_length, input_size,
                                    hidden_size, target_size, vocab_size)

    ids = tf.placeholder(tf.float32, [batch_size, 1], "ids")
    targets = tf.placeholder(tf.float32, [batch_size, target_size], "targets")
    loss = tf.nn.softmax_cross_entropy_with_logits(model, targets)   # targets: labels (e.g. pos/neg/neutral)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    batcher = BatchBucketSampler(data, batch_size)
    acc_batcher = BatchBucketSampler(data, batch_size)

    placeholders += [ids]
    placeholders += [targets]

    hooks = [
        SpeedHook(iteration_interval=50, batch_size=batch_size),
        SaveModelHook(path="../out/save", at_epoch=5),
        #LoadModelHook("./out/save/", 10),
        AccuracyHook(acc_batcher, placeholders, 5),
        LossHook(iteration_interval=10)
    ]

    trainer = Trainer(optimizer, max_epochs, hooks)
    trainer(batcher, placeholders=placeholders, loss=loss, model=model)


    print("Applying to test data, getting predictions for NONE/AGAINST/FAVOR")
    path = "../out/save/latest"

    pad_nr = batch_size-(len(labels_test)%batch_size)+1  # since train/test batches need to be the same size, add padding for test

    data_test = [np.lib.pad(np.asarray(tweets_test), ((0,pad_nr), (0,0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(targets_test), ((0,pad_nr), (0,0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(ids_test), ((0,pad_nr), (0,0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(labels_test), ((0,pad_nr), (0,0)), 'constant', constant_values=(0))
    ]

    corpus_test_batch = BatchBucketSampler(data_test, batch_size)
    predictions_detailed_all = []
    predictions_all = []
    ids_all = []

    with tf.Session() as sess:

        load_model(sess, path)

        total = 0
        correct = 0
        for values in corpus_test_batch:
            total += len(values[-1])
            feed_dict = {}
            for i in range(0, len(placeholders)):
                feed_dict[placeholders[i]] = values[i]
            truth = np.argmax(values[-1], 1)  # values[2] is a 3-legth one-hot vector containing the labels. this is to transform those back into integers
            predictions = sess.run(tf.nn.softmax(model), feed_dict=feed_dict)
            predictions_detailed_all.extend(predictions)
            ids_all.extend(values[-2])
            predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1),
                                     feed_dict=feed_dict)
            predictions_all.extend(predicted)
            correct += sum(truth == predicted)
            #print("pred: ", sess.run(tf.nn.softmax(model), feed_dict=feed_dict))
            #print("ids: ", values[-2])
            print("Num testing samples " + str(total) +
                  "\tAcc " + str(float(correct)/total) +
                  "\tCorrect " + str(correct) + "\tTotal " + str(total))

    return predictions_all, predictions_detailed_all, ids_all


if __name__ == '__main__':

    tweets, targets, labels, ids = reader.readTweetsOfficial("../data/semeval2016-task6-train+dev.txt")
    tweet_tokens = tokenise_tweets(tweets)
    target_tokens = tokenise_tweets(targets)
    count, dictionary, reverse_dictionary = build_dataset([token for senttoks in tweet_tokens+target_tokens for token in senttoks])  #flatten tweets for vocab construction
    transformed_tweets = [transform_tweet(dictionary, senttoks) for senttoks in tweet_tokens]
    transformed_targets = [transform_tweet(dictionary, senttoks) for senttoks in target_tokens]
    transformed_labels = transform_labels(labels)

    tweets_test, targets_test, labels_test, ids_test = reader.readTweetsOfficial("../data/SemEval2016-Task6-subtaskB-testdata-gold.txt")
    tweet_tokens_test = tokenise_tweets(tweets_test)
    target_tokens_test = tokenise_tweets(targets_test)
    transformed_tweets_test = [transform_tweet(dictionary, senttoks) for senttoks in tweet_tokens_test]
    transformed_targets_test = [transform_tweet(dictionary, senttoks) for senttoks in target_tokens_test]
    transformed_labels_test = transform_labels(labels_test)

    predictions_all, predictions_detailed_all, ids_all = test_trainer(dictionary, transformed_tweets, transformed_targets, transformed_labels, ids, transformed_tweets_test,
                 transformed_targets_test, transformed_labels_test, ids_test)

    writer.printPredsToFileByID("../data/SemEval2016-Task6-subtaskB-testdata-gold.txt", "../out/results_subtaskB.txt", ids_all, predictions_all)
    writer.eval("../data/SemEval2016-Task6-subtaskB-testdata-gold.txt", "../out/results_subtaskB.txt")