import tensorflow as tf
import numpy as np

from tfrnn.rnn import Encoder, Projector, rnn_cell
from tfrnn.hooks import SaveModelHook, SaveModelHookDev, AccuracyHook, LossHook, SpeedHook
from tfrnn.batcher import BatchBucketSampler
from tfrnn.util import sample_one_hot, debug_node, load_model, load_model_dev
from tfrnn.hooks import LoadModelHook
from readwrite import reader, writer
from preprocess import tokenise_tweets, transform_targets, transform_tweet, transform_labels, istargetInTweet
#from tensorflow.models.embedding import word2vec
from gensim.models import word2vec, Phrases
from sklearn.metrics import classification_report
from tfrnn.hooks import Hook

class SemEvalHook(Hook):
    """
    Evaluting P/R/F on dev data while training
    """
    def __init__(self, batcher, placeholders, at_every_epoch):
        self.batcher = batcher
        self.placeholders = placeholders
        self.at_every_epoch = at_every_epoch

    def __call__(self, sess, epoch, iteration, model, loss):
        if iteration == 0 and epoch % self.at_every_epoch == 0:
            total = 0
            correct = 0
            truth_all = []
            pred_all = []
            for values in self.batcher:
                total += len(values[-1])
                feed_dict = {}
                for i in range(0, len(self.placeholders)):
                    feed_dict[self.placeholders[i]] = values[i]
                truth = np.argmax(values[-1], 1)  # values[2], batch sampled from data[2], is a 3-legth one-hot vector containing the labels. this is to transform those back into integers
                predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1),
                                     feed_dict=feed_dict)
                correct += sum(truth == predicted)
                truth_all.extend(truth)
                pred_all.extend(predicted)
            print(classification_report(truth_all, pred_all, target_names=["NEUTRAL", "AGAINST", "FAVOR"], digits=4)) #, target_names=[0, 1, 2]))


class Trainer(object):
    """
    Object representing a TensorFlow trainer.
    """

    def __init__(self, optimizer, max_epochs, hooks):
        self.loss = None
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.hooks = hooks

    def __call__(self, batcher, placeholders, loss, pretrain, embedd, model=None, session=None):
        self.loss = loss
        minimization_op = self.optimizer.minimize(loss)
        close_session_after_training = False
        if session is None:
            session = tf.Session()
            close_session_after_training = True  # no session existed before, we provide a temporary session

        init = tf.initialize_all_variables()

        if pretrain == "pre" or pretrain == "pre_cont": # hack if we want to use pre-trained embeddings
            vars = tf.all_variables()
            emb_var = vars[0]
            session.run(emb_var.assign(embedd))

        session.run(init)
        epoch = 1
        while epoch < self.max_epochs:
            iteration = 1
            for values in batcher:
                iteration += 1
                feed_dict = {}
                for i in range(0, len(placeholders)):
                    feed_dict[placeholders[i]] = values[i]
                _, current_loss = session.run([minimization_op, loss], feed_dict=feed_dict)
                current_loss = sum(current_loss)
                for hook in self.hooks:
                    hook(session, epoch, iteration, model, current_loss)

            # calling post-epoch hooks
            for hook in self.hooks:
                hook(session, epoch, 0, model, 0)
            epoch += 1

        if close_session_after_training:
            session.close()



def get_model_conditional(batch_size, max_seq_length, input_size, hidden_size, target_size,
                          vocab_size, pretrain):

    # batch_size x max_seq_length
    inputs = tf.placeholder(tf.int32, [batch_size, max_seq_length])
    inputs_cond = tf.placeholder(tf.int32, [batch_size, max_seq_length])

    cont_train = True
    if pretrain == "pre": # continue training embeddings or not. Currently works better to continue training them.
        cont_train = False
    embedding_matrix = tf.Variable(tf.random_normal([vocab_size, input_size]),  #input_size is embeddings size
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



def get_model_aggr(batch_size, max_seq_length, input_size, hidden_size, target_size,
                         vocab_size, pretrain):

    # batch_size x max_seq_length
    inputs = tf.placeholder(tf.int32, [batch_size, max_seq_length])
    inputs_cond = tf.placeholder(tf.int32, [batch_size, max_seq_length])

    cont_train = True
    if pretrain == "pre":
        cont_train = False
    embedding_matrix = tf.Variable(tf.random_normal([vocab_size, input_size]),  # input_size is embeddings size
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

    lstm_encoder = Encoder(rnn_cell.BasicLSTMCell, input_size, hidden_size)
    start_state = tf.zeros([batch_size, lstm_encoder.state_size])

    # [h_i], [h_i, c_i] <-- LSTM
    # [h_i], [h_i] <-- RNN
    outputs, states = lstm_encoder(inputs_list, start_state, "LSTM")

    model = Projector(target_size, non_linearity=tf.nn.tanh)(outputs[-1])

    return model, [inputs, inputs_cond]



def get_model_tweetonly(batch_size, max_seq_length, input_size, hidden_size, target_size,
                       vocab_size, pretrain):


    # batch_size x max_seq_length
    inputs = tf.placeholder(tf.int32, [batch_size, max_seq_length])

    cont_train = True
    if pretrain == "pre":
        cont_train = False
    embedding_matrix = tf.Variable(tf.random_normal([vocab_size, input_size]),  # input_size is embeddings size
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

    model = Projector(target_size, non_linearity=tf.nn.tanh)(outputs[-1])

    return model, [inputs]






def test_trainer(w2vmodel, tweets, targets, labels, ids, tweets_test, targets_test, labels_test, ids_test, targetInTweet={}, testid = "tweetonly-1", pretrain = "pre_cont"):
    # parameters
    num_samples = 5628
    max_epochs = 21  # 100
    learning_rate = 0.01
    batch_size = 97#101 for with Clinton  # number training examples per training epoch
    input_size = 91
    hidden_size = 60  # making this smaller to avoid overfitting, example is 83
    #pretrain = "pre_cont"  # nopre, pre, pre_cont  : nopre: embeddings are initialised randomly,
                           # pre: word2vec model is loaded, pre_cont: word2vec is loaded and further trained
    aggregated = False
    tweetonly = True
    outfolder = "_".join(["input-" + str(input_size), "hidden-" + str(hidden_size), pretrain, testid])

    # real data stance-semeval
    target_size = 3
    max_seq_length = len(tweets[0])
    #vocab_size = len(dictionary)
    data = [np.asarray(tweets), np.asarray(targets), np.asarray(ids), np.asarray(labels)]
    print("Number training examples:", len(tweets))

    X = w2vmodel.syn0
    vocab_size = len(w2vmodel.vocab)

    if aggregated == True:
        model, placeholders = get_model_aggr(batch_size, max_seq_length, input_size,
                                             hidden_size, target_size, vocab_size, pretrain)
    elif tweetonly == True:
        model, placeholders = get_model_tweetonly(batch_size, max_seq_length, input_size,
                                             hidden_size, target_size, vocab_size, pretrain)
        data = [np.asarray(tweets), np.asarray(ids), np.asarray(labels)]
    else:
        # output of get_model(): model, [inputs, inputs_cond]
        model, placeholders = get_model_conditional(batch_size, max_seq_length, input_size,
                                                    hidden_size, target_size, vocab_size, pretrain)

    ids = tf.placeholder(tf.float32, [batch_size, 1], "ids")
    targets = tf.placeholder(tf.float32, [batch_size, target_size], "targets")
    loss = tf.nn.softmax_cross_entropy_with_logits(model, targets)   # targets: labels (e.g. pos/neg/neutral)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    batcher = BatchBucketSampler(data, batch_size)
    acc_batcher = BatchBucketSampler(data, batch_size)

    placeholders += [ids]
    placeholders += [targets]

    pad_nr = batch_size - (
    len(labels_test) % batch_size) + 1  # since train/test batches need to be the same size, add padding for test

    # prepare the testing data. Needs to be padded to fit the bath size.
    data_test = [np.lib.pad(np.asarray(tweets_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(targets_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(ids_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(labels_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0))
                 ]
    if tweetonly == True:
        data_test = [np.lib.pad(np.asarray(tweets_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(ids_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(labels_test), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0))
                 ]

    corpus_test_batch = BatchBucketSampler(data_test, batch_size)

    hooks = [
        SpeedHook(iteration_interval=50, batch_size=batch_size),
        SaveModelHookDev(path="../out/save/" + outfolder, at_every_epoch=2), #SaveModelHook(path="../out/save", at_epoch=10, at_every_epoch=2),
        #LoadModelHook("./out/save/", 10),
        AccuracyHook(acc_batcher, placeholders, 5),
        SemEvalHook(corpus_test_batch, placeholders, 2),
        LossHook(iteration_interval=50)
    ]

    trainer = Trainer(optimizer, max_epochs, hooks)
    trainer(batcher=batcher, pretrain=pretrain, embedd=X, placeholders=placeholders, loss=loss, model=model)


    print("Applying to test data, getting predictions for NONE/AGAINST/FAVOR")
    #path = "../out/save/latest"

    predictions_detailed_all = []
    predictions_all = []
    ids_all = []

    with tf.Session() as sess:
        load_model_dev(sess, "../out/save/" + outfolder + "_ep20", "model.tf")

        total = 0
        correct = 0
        for values in corpus_test_batch:
            total += len(values[-1])
            feed_dict = {}
            for i in range(0, len(placeholders)):
                feed_dict[placeholders[i]] = values[i]
            truth = np.argmax(values[-1], 1)  # values[2] is a 3-legth one-hot vector containing the labels. this is to transform those back into integers
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
            #print("pred: ", sess.run(tf.nn.softmax(model), feed_dict=feed_dict))
            #print("ids: ", values[-2])
            print("Num testing samples " + str(total) +
                  "\tAcc " + str(float(correct)/total) +
                  "\tCorrect " + str(correct) + "\tTotal " + str(total))


        # potentially do postprocessing
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
                if inTwe == True: #NONE/AGAINST/FAVOUR
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



def readInputAndEval(outfile, stopwords="all", postprocess=True, shortenTargets=False, useAutoTrump=False, useClinton=True, testSetting=True):
    """
    Reading input files, calling the trainer for training the model, evaluate with official script
    :param outfile: name for output file
    :param stopwords: how to filter stopwords, see preprocess.filterStopwords()
    :param postprocess: force against/favor for tweets which contain the target
    :param shortenTargets: shorten the target text, see preprocess.transform_targets()
    :param useAutoTrump: use automatically annotated Trump tweets - not helping, would probably need more work, so not used for best results
    :param useClinton: add the Hillary Clinton dev data to train data
    :param testSetting: evaluate on Trump
    """
    # phrasemodel = Phrases.load("../out/phrase_all.model")
    target = "clinton"

    w2vmodel = word2vec.Word2Vec.load("../out/skip_nostop_single_91features_5minwords_5context")  # without multi is the one without phrases processing

    if testSetting == True:
        trainingdata = "../data/semeval2016-task6-train+dev.txt"
        testdata = "../data/SemEval2016-Task6-subtaskB-testdata-gold.txt"
        target = "trump"
    else:
        trainingdata = "../data/semeval2016-task6-trainingdata_new.txt"
        testdata = "../data/semEval2016-task6-trialdata_new.txt"
    if useClinton == False:
        trainingdata = "../data/semeval2016-task6-trainingdata_new.txt"

    tweets, targets, labels, ids = reader.readTweetsOfficial(trainingdata)

    if useAutoTrump == True:
        tweets_devaut, targets_devaut, labels_devaut, ids_devaut = reader.readTweetsOfficial("../data/semeval2016-task6-autotrump.txt",
                                                                                         encoding='utf-8')

        ids_new = []
        for i in ids_devaut:
            ids_new.append(i + 10000)

        tweets = tweets+tweets_devaut
        targets = targets+targets_devaut
        labels = labels+labels_devaut
        ids = ids+ids_new

    tweet_tokens = tokenise_tweets(tweets, stopwords)  # phrasemodel[tokenise_tweets(tweets)]
    if shortenTargets == False:
        target_tokens = tokenise_tweets(targets, stopwords)  #phrasemodel[tokenise_tweets(transform_targets(targets))]
    else:
        target_tokens = tokenise_tweets(transform_targets(targets), stopwords)

    transformed_tweets = [transform_tweet(w2vmodel, senttoks) for senttoks in tweet_tokens]
    transformed_targets = [transform_tweet(w2vmodel, senttoks) for senttoks in target_tokens]
    transformed_labels = transform_labels(labels)

    tweets_test, targets_test, labels_test, ids_test = reader.readTweetsOfficial(testdata)

    tweet_tokens_test = tokenise_tweets(tweets_test, stopwords)  # phrasemodel[tokenise_tweets(tweets_test)]
    if shortenTargets == False:
        target_tokens_test = tokenise_tweets(targets_test, stopwords)  #phrasemodel[tokenise_tweets(transform_targets(targets_test))]
    else:
        target_tokens_test = tokenise_tweets(transform_targets(targets_test), stopwords)  # #phrasemodel[tokenise_tweets(transform_targets(targets_test))]

    transformed_tweets_test = [transform_tweet(w2vmodel, senttoks) for senttoks in tweet_tokens_test]
    transformed_targets_test = [transform_tweet(w2vmodel, senttoks) for senttoks in target_tokens_test]
    transformed_labels_test = transform_labels(labels_test)

    targetInTweet = {}
    if postprocess == True:
        ids_test_list = [item for sublist in [l.tolist() for l in ids_test] for item in sublist]
        #ids_test_list = [l.tolist() for l in ids_test]
        id_tweet_dict = dict(zip(ids_test_list, tweets_test))
        targetInTweet = istargetInTweet(id_tweet_dict, target)

    predictions_all, predictions_detailed_all, ids_all = test_trainer(w2vmodel, transformed_tweets, transformed_targets,
                                                                      transformed_labels, ids, transformed_tweets_test,
                                                                      transformed_targets_test, transformed_labels_test,
                                                                      ids_test, targetInTweet=targetInTweet)


    writer.printPredsToFileByID(testdata, outfile, ids_all, predictions_all)
    writer.eval(testdata, outfile)


if __name__ == '__main__':
    readInputAndEval("../out/results_subtaskB_most_postpr_tweetonly_test3.txt", stopwords="punctonly", postprocess=True, testSetting=True, useClinton=True)
