#import sys
#sys.path.append("/path/to/stance-conditional")

import tensorflow as tf
import numpy as np

from stancedetection.rnn import Encoder, Projector, Hook, AccuracyHook, LossHook, SpeedHook, BatchBucketSampler, TraceHook, SaveModelHookDev
from stancedetection.rnn import Trainer, SemEvalHook, AccuracyHookIgnoreNeutral, load_model_dev
from readwrite import reader, writer
from preprocess import tokenise_tweets, transform_targets, transform_tweet, transform_labels, istargetInTweet, istargetInTweetSing
from gensim.models import word2vec, Phrases
import os
from tensorflow.models.rnn import rnn_cell


def get_model_conditional(batch_size, max_seq_length, input_size, hidden_size, target_size,
                          vocab_size, pretrain, tanhOrSoftmax, dropout):
    """
    Unidirectional conditional encoding model
    :param pretrain:  "pre": use pretrained word embeddings, "pre-cont": use pre-trained embeddings and continue training them, otherwise: random initialisation
    """

    # batch_size x max_seq_length
    inputs = tf.placeholder(tf.int32, [batch_size, max_seq_length])
    inputs_cond = tf.placeholder(tf.int32, [batch_size, max_seq_length])

    cont_train = True
    if pretrain == "pre": # continue training embeddings or not. Currently works better to continue training them.
        cont_train = False
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, input_size], -0.1, 0.1),  #input_size is embeddings size
                                   name="embedding_matrix", trainable=cont_train)

    # batch_size x max_seq_length x input_size
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)
    embedded_inputs_cond = tf.nn.embedding_lookup(embedding_matrix, inputs_cond)

    # [batch_size x inputs_size] with max_seq_length elements
    # fixme: possibly inefficient
    # inputs_list[0]: batch_size x input[0] <-- word vector of the first word
    inputs_list = [tf.squeeze(x) for x in
                   tf.split(1, max_seq_length, embedded_inputs)]
    inputs_cond_list = [tf.squeeze(x) for x in
                        tf.split(1, max_seq_length, embedded_inputs_cond)]

    drop_prob = None
    if dropout:
        drop_prob = 0.1
    lstm_encoder = Encoder(rnn_cell.BasicLSTMCell, input_size, hidden_size, drop_prob, drop_prob)

    start_state = tf.zeros([batch_size, lstm_encoder.state_size])

    # [h_i], [h_i, c_i] <-- LSTM
    # [h_i], [h_i] <-- RNN
    outputs, states = lstm_encoder(inputs_list, start_state, "LSTM")

    # running a second LSTM conditioned on the last state of the first
    outputs_cond, states_cond = lstm_encoder(inputs_cond_list, states[-1],
                                             "LSTMcond")

    outputs_fin = outputs_cond[-1]
    if tanhOrSoftmax == "tanh":
        model = Projector(target_size, non_linearity=tf.nn.tanh, bias=True)(outputs_fin) #tf.nn.softmax
    else:
        model = Projector(target_size, non_linearity=tf.nn.softmax, bias=True)(outputs_fin)  # tf.nn.softmax

    return model, [inputs, inputs_cond]



def get_model_concat(batch_size, max_seq_length, input_size, hidden_size, target_size,
                     vocab_size, pretrain, tanhOrSoftmax, dropout):
    """
    LSTM over target and over tweet, concatenated
    :param pretrain:  "pre": use pretrained word embeddings, "pre-cont": use pre-trained embeddings and continue training them, otherwise: random initialisation
    """

    # batch_size x max_seq_length
    inputs = tf.placeholder(tf.int32, [batch_size, max_seq_length])
    inputs_cond = tf.placeholder(tf.int32, [batch_size, max_seq_length])

    cont_train = True
    if pretrain == "pre":
        cont_train = False
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, input_size], -0.1, 0.1),  # input_size is embeddings size
                               name="embedding_matrix", trainable=cont_train)

    # batch_size x max_seq_length x input_size
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)
    embedded_inputs_cond = tf.nn.embedding_lookup(embedding_matrix, inputs_cond)

    embedded_inputs_all = tf.concat(1, [embedded_inputs, embedded_inputs_cond])   # concatenating the two embeddings

    # [batch_size x inputs_size] with max_seq_length elements
    # fixme: possibly inefficient
    # inputs_list[0]: batch_size x input[0] <-- word vector of the first word
    inputs_list = [tf.squeeze(x) for x in
               tf.split(1, max_seq_length*2, embedded_inputs_all)]


    drop_prob = None
    if dropout:
        drop_prob = 0.1
    lstm_encoder = Encoder(rnn_cell.BasicLSTMCell, input_size, hidden_size, drop_prob, drop_prob)


    start_state = tf.zeros([batch_size, lstm_encoder.state_size])

    # [h_i], [h_i, c_i] <-- LSTM
    # [h_i], [h_i] <-- RNN
    outputs, states = lstm_encoder(inputs_list, start_state, "LSTM")

    outputs_fin = outputs[-1]

    if tanhOrSoftmax == "tanh":
        model = Projector(target_size, non_linearity=tf.nn.tanh)(outputs_fin) #tf.nn.softmax
    else:
        model = Projector(target_size, non_linearity=tf.nn.softmax)(outputs_fin)  # tf.nn.softmax


    return model, [inputs, inputs_cond]



def get_model_tweetonly(batch_size, max_seq_length, input_size, hidden_size, target_size,
                       vocab_size, pretrain, tanhOrSoftmax, dropout):
    """
    LSTM over tweet only
    :param pretrain:  "pre": use pretrained word embeddings, "pre-cont": use pre-trained embeddings and continue training them, otherwise: random initialisation
    """

    # batch_size x max_seq_length
    inputs = tf.placeholder(tf.int32, [batch_size, max_seq_length])

    cont_train = True
    if pretrain == "pre":
        cont_train = False
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, input_size], -0.1, 0.1),  # input_size is embeddings size
                               name="embedding_matrix", trainable=cont_train)

    # batch_size x max_seq_length x input_size
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)


    # [batch_size x inputs_size] with max_seq_length elements
    # fixme: possibly inefficient
    # inputs_list[0]: batch_size x input[0] <-- word vector of the first word
    inputs_list = [tf.squeeze(x) for x in
               tf.split(1, max_seq_length, embedded_inputs)]

    lstm_encoder = Encoder(rnn_cell.BasicLSTMCell, input_size, hidden_size)
    start_state = tf.zeros([batch_size, lstm_encoder.state_size])

    # [h_i], [h_i, c_i] <-- LSTM
    # [h_i], [h_i] <-- RNN
    outputs, states = lstm_encoder(inputs_list, start_state, "LSTM")

    drop_prob = None
    if dropout:
        drop_prob = 0.1

    lstm_encoder = Encoder(rnn_cell.BasicLSTMCell, input_size, hidden_size, drop_prob, drop_prob)

    outputs_fin = outputs[-1]
    if tanhOrSoftmax == "tanh":
        model = Projector(target_size, non_linearity=tf.nn.tanh)(outputs_fin) #tf.nn.softmax
    else:
        model = Projector(target_size, non_linearity=tf.nn.softmax)(outputs_fin)  # tf.nn.softmax


    return model, [inputs]



def get_model_bidirectional_conditioning(batch_size, max_seq_length, input_size, hidden_size, target_size,
                                         vocab_size, pretrain, tanhOrSoftmax, dropout):
    """
    Bidirectional conditioning model
    :param pretrain:  "pre": use pretrained word embeddings, "pre-cont": use pre-trained embeddings and continue training them, otherwise: random initialisation
    """

    # batch_size x max_seq_length
    inputs = tf.placeholder(tf.int32, [batch_size, max_seq_length])
    inputs_cond = tf.placeholder(tf.int32, [batch_size, max_seq_length])

    cont_train = True
    if pretrain == "pre":  # continue training embeddings or not. Currently works better to continue training them.
        cont_train = False
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, input_size], -0.1, 0.1),  # input_size is embeddings size
                               name="embedding_matrix", trainable=cont_train)

    # batch_size x max_seq_length x input_size
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)
    embedded_inputs_cond = tf.nn.embedding_lookup(embedding_matrix, inputs_cond)

    # [batch_size x inputs_size] with max_seq_length elements
    # fixme: possibly inefficient
    # inputs_list[0]: batch_size x input[0] <-- word vector of the first word
    inputs_list = [tf.squeeze(x) for x in
               tf.split(1, max_seq_length, embedded_inputs)]
    inputs_cond_list = [tf.squeeze(x) for x in
                    tf.split(1, max_seq_length, embedded_inputs_cond)]

    drop_prob = None
    if dropout:
        drop_prob = 0.1
    lstm_encoder = Encoder(rnn_cell.BasicLSTMCell, input_size, hidden_size, drop_prob, drop_prob)

    start_state = tf.zeros([batch_size, lstm_encoder.state_size])

    ### FORWARD

    # [h_i], [h_i, c_i] <-- LSTM
    # [h_i], [h_i] <-- RNN
    fw_outputs, fw_states = lstm_encoder(inputs_list, start_state, "LSTM")

    # running a second LSTM conditioned on the last state of the first
    fw_outputs_cond, fw_states_cond = lstm_encoder(inputs_cond_list, fw_states[-1],
                                               "LSTMcond")

    fw_outputs_fin = fw_outputs_cond[-1]

    ### BACKWARD
    bw_outputs, bw_states = lstm_encoder(inputs_list[::-1], start_state, "LSTM_bw")
    bw_outputs_cond, bw_states_cond = lstm_encoder(inputs_cond_list[::-1], bw_states[-1],
                                               "LSTMcond_bw")
    bw_outputs_fin = bw_outputs_cond[-1]

    outputs_fin = tf.concat(1, [fw_outputs_fin, bw_outputs_fin])


    if tanhOrSoftmax == "tanh":
        model = Projector(target_size, non_linearity=tf.nn.tanh, bias=True)(outputs_fin)  # tf.nn.softmax
    else:
        model = Projector(target_size, non_linearity=tf.nn.softmax, bias=True)(outputs_fin)  # tf.nn.softmax

    return model, [inputs, inputs_cond]



def get_model_conditional_target_feed(batch_size, max_seq_length, input_size, hidden_size, target_size,
                                      vocab_size, pretrain, tanhOrSoftmax, dropout):
    """
    Experimental, feed target during tweet processing
    :param pretrain:  "pre": use pretrained word embeddings, "pre-cont": use pre-trained embeddings and continue training them, otherwise: random initialisation
    """
    # batch_size x max_seq_length
    inputs = tf.placeholder(tf.int32, [batch_size, max_seq_length])
    inputs_cond = tf.placeholder(tf.int32, [batch_size, max_seq_length])

    cont_train = True
    if pretrain == "pre":  # continue training embeddings or not. Currently works better to continue training them.
        cont_train = False
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, input_size], -0.1, 0.1),
                                   # input_size is embeddings size
                                   name="embedding_matrix", trainable=cont_train)

    # batch_size x max_seq_length x input_size
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)
    embedded_inputs_cond = tf.nn.embedding_lookup(embedding_matrix, inputs_cond)

    # [batch_size x inputs_size] with max_seq_length elements
    # fixme: possibly inefficient
    # inputs_list[0]: batch_size x input[0] <-- word vector of the first word
    inputs_list = [tf.squeeze(x) for x in
                   tf.split(1, max_seq_length, embedded_inputs)]

    drop_prob = None
    if dropout:
        drop_prob = 0.1
    lstm_encoder_target = Encoder(rnn_cell.BasicLSTMCell, input_size, hidden_size, drop_prob, drop_prob)

    start_state = tf.zeros([batch_size, lstm_encoder_target.state_size])

    # [h_i], [h_i, c_i] <-- LSTM
    # [h_i], [h_i] <-- RNN
    outputs, states = lstm_encoder_target(inputs_list, start_state, "LSTM")

    lstm_encoder_tweet = Encoder(rnn_cell.BasicLSTMCell, input_size + 2 * hidden_size, hidden_size, drop_prob,
                                 drop_prob)

    inputs_cond_list = [tf.concat(1, [tf.squeeze(x), states[-1]]) for x in
                        tf.split(1, max_seq_length, embedded_inputs_cond)]

    # running a second LSTM conditioned on the last state of the first
    outputs_cond, states_cond = lstm_encoder_tweet(inputs_cond_list, states[-1],
                                                   "LSTMcond")

    outputs_fin = outputs_cond[-1]


    if tanhOrSoftmax == "tanh":
        model = Projector(target_size, non_linearity=tf.nn.tanh, bias=True)(outputs_fin)  # tf.nn.softmax
    else:
        model = Projector(target_size, non_linearity=tf.nn.softmax, bias=True)(outputs_fin)  # tf.nn.softmax

    return model, [inputs, inputs_cond]



def get_model_bicond_sepembed(batch_size, max_seq_length, input_size, hidden_size, target_size,
                              vocab_size, pretrain, tanhOrSoftmax, dropout):
    """
    Bidirectional conditional encoding with separate embeddings matrices for tweets and targets lookup
    :param pretrain:  "pre": use pretrained word embeddings, "pre-cont": use pre-trained embeddings and continue training them, otherwise: random initialisation
    """

    # batch_size x max_seq_length
    inputs = tf.placeholder(tf.int32, [batch_size, max_seq_length])
    inputs_cond = tf.placeholder(tf.int32, [batch_size, max_seq_length])

    cont_train = True
    if pretrain == "pre":  # continue training embeddings or not. Currently works better to continue training them.
        cont_train = False
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, input_size], -0.1, 0.1),  # input_size is embeddings size
                               name="embedding_matrix", trainable=cont_train)
    embedding_matrix_cond = tf.Variable(tf.random_uniform([vocab_size, input_size], -0.1, 0.1),
                                name="embedding_matrix", trainable=cont_train)


    # batch_size x max_seq_length x input_size
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)
    embedded_inputs_cond = tf.nn.embedding_lookup(embedding_matrix_cond, inputs_cond)


    # [batch_size x inputs_size] with max_seq_length elements
    # fixme: possibly inefficient
    # inputs_list[0]: batch_size x input[0] <-- word vector of the first word
    inputs_list = [tf.squeeze(x) for x in
               tf.split(1, max_seq_length, embedded_inputs)]
    inputs_cond_list = [tf.squeeze(x) for x in
                    tf.split(1, max_seq_length, embedded_inputs_cond)]

    drop_prob = None
    if dropout:
        drop_prob = 0.1
    lstm_encoder = Encoder(rnn_cell.BasicLSTMCell, input_size, hidden_size, drop_prob, drop_prob)

    start_state = tf.zeros([batch_size, lstm_encoder.state_size])

    ### FORWARD

    # [h_i], [h_i, c_i] <-- LSTM
    # [h_i], [h_i] <-- RNN
    fw_outputs, fw_states = lstm_encoder(inputs_list, start_state, "LSTM")

    # running a second LSTM conditioned on the last state of the first
    fw_outputs_cond, fw_states_cond = lstm_encoder(inputs_cond_list, fw_states[-1],
                                               "LSTMcond")

    fw_outputs_fin = fw_outputs_cond[-1]

    ### BACKWARD
    bw_outputs, bw_states = lstm_encoder(inputs_list[::-1], start_state, "LSTM_bw")
    bw_outputs_cond, bw_states_cond = lstm_encoder(inputs_cond_list[::-1], bw_states[-1],
                                               "LSTMcond_bw")
    bw_outputs_fin = bw_outputs_cond[-1]

    outputs_fin = tf.concat(1, [fw_outputs_fin, bw_outputs_fin])


    if tanhOrSoftmax == "tanh":
        model = Projector(target_size, non_linearity=tf.nn.tanh, bias=True)(outputs_fin)  # tf.nn.softmax
    else:
        model = Projector(target_size, non_linearity=tf.nn.softmax, bias=True)(outputs_fin)  # tf.nn.softmax

    return model, [inputs, inputs_cond]





def test_trainer(testsetting, w2vmodel, tweets, targets, labels, ids, tweets_test, targets_test, labels_test, ids_test, hidden_size, max_epochs, tanhOrSoftmax, dropout, modeltype="conditional", targetInTweet={}, testid = "test-1", pretrain = "pre_cont", acc_thresh=0.9, sep = False):
    """
    Method for creating the different models and training them
    :param testsetting: "True" for SemEval test setting (Donald Trump), "False" for dev setting (Hillary Clinton)
    :param w2vmodel: location of word2vec model
    :param tweets: training tweets, read and converted in readInputAndEval()
    :param targets: training targets, read and converted in readInputAndEval()
    :param labels: training labels, read and converted in readInputAndEval()
    :param ids: ids of training instances
    :param tweets_test: testing tweets, read and converted in readInputAndEval()
    :param targets_test: testing targets, read and converted in readInputAndEval()
    :param labels_test: testing labels, read and converted in readInputAndEval()
    :param ids_test: ids of testing instances
    :param hidden_size: size of hidden layer
    :param max_epochs: maximum number of training epochs
    :param tanhOrSoftmax: tanh or softmax in projector
    :param dropout: use dropout or not
    :param modeltype: "concat", "tweetonly", "conditional", "conditional-reverse", "bicond", "conditional-target-feed", "bicond-sepembed"
    :param targetInTweet: dictionary produced with id to targetInTweet mappings in readInputAndEval(), used for postprocessing
    :param testid: id of test run
    :param pretrain: "pre" (use pretrained word embeddings), "pre_cont" (use pretrained word embeddings and continue training them), "random" (random word embeddings initialisations)
    :param acc_thresh: experimental, stop training at certain accuracy threshold (between 0 and 1)
    :param sep: True for using separate embeddings matrices, false for one (default)
    :return:
    """

    # parameters
    learning_rate = 0.0001
    batch_size = 70
    input_size = 100

    outfolder = "_".join([testid, modeltype, testsetting, "hidden-" + str(hidden_size), tanhOrSoftmax])

    # real data stance-semeval
    target_size = 3
    max_seq_length = len(tweets[0])
    if modeltype == "conditional-reverse":
        data = [np.asarray(targets), np.asarray(tweets), np.asarray(ids), np.asarray(labels)]
    else:
        data = [np.asarray(tweets), np.asarray(targets), np.asarray(ids), np.asarray(labels)]

    X = w2vmodel.syn0
    vocab_size = len(w2vmodel.vocab)

    if modeltype == "concat":
        model, placeholders = get_model_concat(batch_size, max_seq_length, input_size,
                                               hidden_size, target_size, vocab_size, pretrain, tanhOrSoftmax, dropout)
    elif modeltype == "tweetonly":
        model, placeholders = get_model_tweetonly(batch_size, max_seq_length, input_size,
                                             hidden_size, target_size, vocab_size, pretrain, tanhOrSoftmax, dropout)
        data = [np.asarray(tweets), np.asarray(ids), np.asarray(labels)]
    elif modeltype == "conditional" or modeltype == "conditional-reverse":
        # output of get_model(): model, [inputs, inputs_cond]
        model, placeholders = get_model_conditional(batch_size, max_seq_length, input_size,
                                            hidden_size, target_size, vocab_size, pretrain, tanhOrSoftmax, dropout)
    elif modeltype == "bicond":
        model, placeholders = get_model_bidirectional_conditioning(batch_size, max_seq_length, input_size, hidden_size, target_size,
                                                                   vocab_size, pretrain, tanhOrSoftmax, dropout)
    elif modeltype == "conditional-target-feed":
        model, placeholders = get_model_conditional_target_feed(batch_size, max_seq_length, input_size, hidden_size,
                                                                target_size,
                                                                vocab_size, pretrain, tanhOrSoftmax, dropout)
    elif modeltype == "bicond-sepembed":
        model, placeholders = get_model_bicond_sepembed(batch_size, max_seq_length, input_size, hidden_size,
                                                        target_size,
                                                        vocab_size, pretrain, tanhOrSoftmax, dropout)
        sep = True

    ids = tf.placeholder(tf.float32, [batch_size, 1], "ids")  #ids are so that the dev/test samples can be recovered later since we shuffle
    targets = tf.placeholder(tf.float32, [batch_size, target_size], "targets")


    loss = tf.nn.softmax_cross_entropy_with_logits(model, targets)   # targets: labels (e.g. pos/neg/neutral)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    batcher = BatchBucketSampler(data, batch_size)
    acc_batcher = BatchBucketSampler(data, batch_size)

    placeholders += [ids]
    placeholders += [targets]

    pad_nr = batch_size - (
    len(labels_test) % batch_size) + 1  # since train/test batches need to be the same size, add padding for test

    # prepare the testing data. Needs to be padded to fit the batch size.
    if modeltype == "tweetonly":
        data_test = [np.lib.pad(np.asarray(tweets_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(ids_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(labels_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0))
                 ]
    elif modeltype == "conditional-reverse":
        data_test = [np.lib.pad(np.asarray(targets_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                     np.lib.pad(np.asarray(tweets_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                     np.lib.pad(np.asarray(ids_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                     np.lib.pad(np.asarray(labels_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0))
                     ]
    else:
        data_test = [np.lib.pad(np.asarray(tweets_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                     np.lib.pad(np.asarray(targets_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                     np.lib.pad(np.asarray(ids_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                     np.lib.pad(np.asarray(labels_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0))
                     ]

    corpus_test_batch = BatchBucketSampler(data_test, batch_size)


    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter("./out/save", graph_def=sess.graph_def)

        hooks = [
            SpeedHook(summary_writer, iteration_interval=50, batch_size=batch_size),
            SaveModelHookDev(path="../out/save/" + outfolder, at_every_epoch=1),
            SemEvalHook(corpus_test_batch, placeholders, 1),
            LossHook(summary_writer, iteration_interval=50),
            AccuracyHook(summary_writer, acc_batcher, placeholders, 2),
            AccuracyHookIgnoreNeutral(summary_writer, acc_batcher, placeholders, 2)
        ]

        trainer = Trainer(optimizer, max_epochs, hooks)
        epoch = trainer(batcher=batcher, acc_thresh=acc_thresh, pretrain=pretrain, embedd=X, placeholders=placeholders,
                        loss=loss, model=model, sep=sep)

        print("Applying to test data, getting predictions for NONE/AGAINST/FAVOR")

        predictions_detailed_all = []
        predictions_all = []
        ids_all = []

        load_model_dev(sess, "../out/save/" + outfolder + "_ep" + str(epoch), "model.tf")

        total = 0
        correct = 0
        for values in corpus_test_batch:
            total += len(values[-1])
            feed_dict = {}
            for i in range(0, len(placeholders)):
                feed_dict[placeholders[i]] = values[i]
            truth = np.argmax(values[-1], 1)  # values[2] is a 3-length one-hot vector containing the labels
            if pretrain == "pre" and sep == True:  # this is a bit hacky. To do: improve
                vars = tf.all_variables()
                emb_var = vars[0]
                emb_var2 = vars[1]
                sess.run(emb_var.assign(X))
                sess.run(emb_var2.assign(X))
            if pretrain == "pre":  # this is a bit hacky. To do: improve
                vars = tf.all_variables()
                emb_var = vars[0]
                sess.run(emb_var.assign(X))
            predictions = sess.run(tf.nn.softmax(model), feed_dict=feed_dict)
            predictions_detailed_all.extend(predictions)
            ids_all.extend(values[-2])
            predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1),
                                     feed_dict=feed_dict)
            predictions_all.extend(predicted)
            correct += sum(truth == predicted)

            print("Num testing samples " + str(total) +
                  "\tAcc " + str(float(correct)/total) +
                  "\tCorrect " + str(correct) + "\tTotal " + str(total))


        # postprocessing
        if targetInTweet != {}:

            predictions_new = []
            ids_new = []
            it = 0
            for pred_prob in predictions_detailed_all:
                id = ids_all[it]
                if id == 0.0:
                    it += 1
                    continue
                inTwe = targetInTweet[id.tolist()[0]]
                if inTwe == True: #and (pred_prob[2] > 0.1 or pred_prob[1] > 0.1): #NONE/AGAINST/FAVOUR
                    #print(str(id), "inTwe!")
                    pred = 1
                    if pred_prob[2] > pred_prob[1]:
                        pred = 2
                    predictions_new.append(pred)
                else:
                    plist = pred_prob.tolist()
                    pred = plist.index(max(plist))
                    predictions_new.append(pred)
                it += 1
                ids_new.append(id)
            return predictions_new, predictions_detailed_all, ids_new

    return predictions_all, predictions_detailed_all, ids_all



def readInputAndEval(testSetting, outfile, hidden_size, max_epochs, tanhOrSoftmax, dropout, stopwords="most", testid="test1", modeltype="bicond", word2vecmodel="small", postprocess=True, shortenTargets=False, useAutoTrump=False, useClinton=True, acc_thresh=1.0, pretrain="pre_cont", usePhrases=False):
    """
    Reading input files, calling the trainer for training the model, evaluate with official script
    :param outfile: name for output file
    :param stopwords: how to filter stopwords, see preprocess.filterStopwords()
    :param postprocess: force against/favor for tweets which contain the target
    :param shortenTargets: shorten the target text, see preprocess.transform_targets()
    :param useAutoTrump: use automatically annotated Trump tweets, experimental, not helping at the moment
    :param useClinton: add the Hillary Clinton dev data to train data
    :param testSetting: evaluate on Trump
    """


    if word2vecmodel == "small":
        w2vmodel = word2vec.Word2Vec.load("../out/skip_nostop_single_100features_5minwords_5context")
    else:
        w2vmodel = word2vec.Word2Vec.load("../out/skip_nostop_single_100features_5minwords_5context_big")

    if usePhrases == True:
        phrasemodel = Phrases.load("../out/phrase_all.model")
        w2vmodel = word2vec.Word2Vec.load("../out/skip_nostop_multi_100features_5minwords_5context")

    if testSetting == "true":
        trainingdata = "../data/semeval2016-task6-train+dev.txt"
        testdata = "../data/SemEval2016-Task6-subtaskB-testdata-gold.txt"
    elif testSetting == "weaklySup":
        trainingdata = "../data/trump_autolabelled.txt"
        testdata = "../data/SemEval2016-Task6-subtaskB-testdata-gold.txt"
        enc = "utf-8"
    else:
        trainingdata = "../data/semeval2016-task6-trainingdata_new.txt"
        testdata = "../data/semEval2016-task6-trialdata_new.txt"
    if useClinton == False:
        trainingdata = "../data/semeval2016-task6-trainingdata_new.txt"

    tweets, targets, labels, ids = reader.readTweetsOfficial(trainingdata, encoding=enc)

    # this is for using automatically labelled Donald Trump data in addition to task data
    if useAutoTrump == True:
        tweets_devaut, targets_devaut, labels_devaut, ids_devaut = reader.readTweetsOfficial("../data/trump_autolabelled.txt",
                                                                                         encoding='utf-8')
        ids_new = []
        for i in ids_devaut:
            ids_new.append(i + 10000)

        tweets = tweets+tweets_devaut
        targets = targets+targets_devaut
        labels = labels+labels_devaut
        ids = ids+ids_new


    if usePhrases == False:
        tweet_tokens = tokenise_tweets(tweets, stopwords)
        if shortenTargets == False:
            target_tokens = tokenise_tweets(targets, stopwords)
        else:
            target_tokens = tokenise_tweets(transform_targets(targets), stopwords)
    else:
        tweet_tokens = phrasemodel[tokenise_tweets(tweets, stopwords)]
        if shortenTargets == False:
            target_tokens = phrasemodel[tokenise_tweets(targets,
                                            stopwords)]
        else:
            target_tokens = phrasemodel[tokenise_tweets(transform_targets(targets), stopwords)]


    transformed_tweets = [transform_tweet(w2vmodel, senttoks) for senttoks in tweet_tokens]
    transformed_targets = [transform_tweet(w2vmodel, senttoks) for senttoks in target_tokens]
    transformed_labels = transform_labels(labels)

    tweets_test, targets_test, labels_test, ids_test = reader.readTweetsOfficial(testdata)

    if usePhrases == False:
        tweet_tokens_test = tokenise_tweets(tweets_test, stopwords)
        if shortenTargets == False:
            target_tokens_test = tokenise_tweets(targets_test, stopwords)
        else:
            target_tokens_test = tokenise_tweets(transform_targets(targets_test), stopwords)
    else:
        tweet_tokens_test = phrasemodel[tokenise_tweets(tweets_test, stopwords)]
        if shortenTargets == False:
            target_tokens_test = phrasemodel[tokenise_tweets(targets_test, stopwords)]
        else:
            target_tokens_test = phrasemodel[tokenise_tweets(transform_targets(targets_test), stopwords)]


    transformed_tweets_test = [transform_tweet(w2vmodel, senttoks) for senttoks in tweet_tokens_test]
    transformed_targets_test = [transform_tweet(w2vmodel, senttoks) for senttoks in target_tokens_test]
    transformed_labels_test = transform_labels(labels_test)

    targetInTweet = {}
    if postprocess == True:
        ids_test_list = [item for sublist in [l.tolist() for l in ids_test] for item in sublist]
        id_tweet_dict = dict(zip(ids_test_list, tweets_test))
        targetInTweet = istargetInTweet(id_tweet_dict, targets_test) #istargetInTweet

    predictions_all, predictions_detailed_all, ids_all = test_trainer(testSetting, w2vmodel, transformed_tweets, transformed_targets, transformed_labels, ids, transformed_tweets_test,
                                                                      transformed_targets_test, transformed_labels_test, ids_test, hidden_size, max_epochs,
                                                                      tanhOrSoftmax, dropout, modeltype, targetInTweet,
                                                                      testid, acc_thresh=acc_thresh, pretrain=pretrain)



    writer.printPredsToFileByID(testdata, outfile, ids_all, predictions_all)
    writer.eval(testdata, outfile, evalscript="eval.pl")


def readResfilesAndEval(testSetting, outfile):

        if testSetting == "true":
            testdata = "../data/SemEval2016-Task6-subtaskB-testdata-gold.txt"
        else:
            testdata = "../data/semEval2016-task6-trialdata_new.txt"

        writer.eval(testdata, outfile, evalscript="eval.pl")


if __name__ == '__main__':
    np.random.seed(1337)
    tf.set_random_seed(1337)

    SINGLE_RUN = False
    EVALONLY = False

    if SINGLE_RUN:
        hidden_size = 100
        max_epochs = 8
        modeltype = "bicond" # this is default
        word2vecmodel = "big"
        stopwords = "most"
        tanhOrSoftmax = "tanh"
        dropout = "true"
        pretrain = "pre_cont" # this is default
        testsetting = "weaklySup"
        testid = "test1"

        outfile = "../out/results_quicktest_" + testsetting + "_" + modeltype + "_" + str(hidden_size) + "_" + dropout + "_" + tanhOrSoftmax + "_" + str(max_epochs) + "_" + testid + ".txt"

        readInputAndEval(testsetting, outfile, hidden_size, max_epochs, tanhOrSoftmax, dropout, stopwords, testid, modeltype, word2vecmodel)

    else:

        # code for testing different combinations below
        hidden_size = [100] #[50, 55, 60]
        #acc_tresh = 1.0
        max_epochs = 8
        w2v = "big" #small
        modeltype = ["bicond"]
        stopwords = ["most"]
        dropout = ["true"]
        testsetting = ["weaklySup"]
        pretrain = ["pre_cont"]

        for i in range(10):
            for modelt in modeltype:
                for drop in dropout:
                    for tests in testsetting:
                        for hid in hidden_size:
                            for pre in pretrain:
                                outfile = "../out/results_batch70_2_morehash3_ep7_9-1e-3-" + tests + "_" + modelt + "_w2v" + w2v + "_hidd" + str(hid) + "_drop" + drop + "_" + pre + "_" + str(i) + ".txt"
                                print(outfile)

                                if EVALONLY == False:
                                    readInputAndEval(tests, outfile, hid, max_epochs, "tanh", drop, "most", str(i), modelt, w2v, acc_thresh=1)
                                    tf.ops.reset_default_graph()
                                else:
                                    readResfilesAndEval(tests, outfile)
