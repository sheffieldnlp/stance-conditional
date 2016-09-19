import tensorflow as tf
import numpy as np
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell, linear, DropoutWrapper, EmbeddingWrapper
from tensorflow.models.rnn.rnn import rnn as rnn_encoder_factory
from tensorflow.models.rnn.seq2seq import rnn_decoder as rnn_decoder_factory
from tensorflow.models.rnn.translate import data_utils
from tensorflow.python.ops import variable_scope as vs
import tensorflow.models.rnn.rnn_cell as rnn_cell
import tensorflow as tf
import numpy as np
import time
import os
import numpy.ma as ma
from sklearn.metrics import classification_report




class Encoder(object):
    """
    Object representing an RNN encoder.
    """

    def __init__(self, cell_factory, input_size, hidden_size, input_dropout=None, output_dropout=None):
        """
        :param cell_factory:
        :param input_size:
        :param hidden_size:
        :return:
        """
        self.cell_factory = cell_factory
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = self.cell_factory(self.hidden_size)
        if input_dropout is not None or output_dropout is not None:
            self.cell = DropoutWrapper(self.cell, 1-(input_dropout or 0.0), 1-(output_dropout or 0.0))
        self.state_size = self.cell.state_size

    def __call__(self, inputs, start_state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        Args:
          inputs: list of 2D Tensors with shape [batch_size x self.input_size].
          start_state: 2D Tensor with shape [batch_size x self.state_size].
          scope: VariableScope for the created subgraph; defaults to class name.
        Returns:
          A pair containing:
          - Outputs: list of 2D Tensors with shape [batch_size x self.output_size]
          - States: list of 2D Tensors with shape [batch_size x self.state_size].
        """
        with vs.variable_scope(scope or "Encoder"):
            return rnn_encoder_factory(self.cell, inputs, start_state)



class Projector(object):
    def __init__(self, to_size, bias=False, non_linearity=None):
        self.to_size = to_size
        self.bias = bias
        self.non_linearity = non_linearity

    def __call__(self, inputs, scope=None):
        """
        :param inputs: list of 2D Tensors with shape [batch_size x self.from_size]
        :return: list of 2D Tensors with shape [batch_size x self.to_size]
        """
        with vs.variable_scope(scope or "Projector"):
            projected = linear(inputs, self.to_size, self.bias)
            if self.non_linearity is not None:
                projected = self.non_linearity(projected)
        return projected



LOSS_TRACE_TAG = "Loss"
SPEED_TRACE_TAG = "Speed"
ACCURACY_TRACE_TAG = "Accuracy"


class Hook(object):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, sess, epoch, iteration, model, loss):
        raise NotImplementedError


class TraceHook(object):
    def __init__(self, summary_writer):
        self.summary_writer = summary_writer

    def __call__(self, sess, epoch, iteration, model, loss):
        raise NotImplementedError

    def update_summary(self, sess, current_step, title, value):
        cur_summary = tf.scalar_summary(title, value)
        merged_summary_op = tf.merge_summary([cur_summary])  # if you are using some summaries, merge them
        summary_str = sess.run(merged_summary_op)
        self.summary_writer.add_summary(summary_str, current_step)


class LossHook(TraceHook):
    def __init__(self, summary_writer, iteration_interval):
        super().__init__(summary_writer)
        self.iteration_interval = iteration_interval
        self.acc_loss = 0

    def __call__(self, sess, epoch, iteration, model, loss):
        self.acc_loss += loss
        if not iteration == 0 and iteration % self.iteration_interval == 0:
            loss = self.acc_loss / self.iteration_interval
            print("Epoch " + str(epoch) +
                  "\tIter " + str(iteration) +
                  "\tLoss " + str(loss))
            self.update_summary(sess, iteration, LOSS_TRACE_TAG, loss)
            self.acc_loss = 0


class SpeedHook(TraceHook):
    def __init__(self, summary_writer, iteration_interval, batch_size):
        super().__init__(summary_writer)
        self.iteration_interval = iteration_interval
        self.batch_size = batch_size
        self.t0 = time.time()
        self.num_examples = iteration_interval * batch_size

    def __call__(self, sess, epoch, iteration, model, loss):
        if not iteration == 0 and iteration % self.iteration_interval == 0:
            diff = time.time() - self.t0
            speed = int(self.num_examples / diff)
            print("Epoch " + str(epoch) +
                  "\tIter " + str(iteration) +
                  "\tExamples/s " + str(speed))
            self.update_summary(sess, iteration, SPEED_TRACE_TAG, float(speed))
            self.t0 = time.time()


class AccuracyHook(TraceHook):
    def __init__(self, summary_writer, batcher, placeholders, at_every_epoch):
        super().__init__(summary_writer)
        self.batcher = batcher
        self.placeholders = placeholders
        self.at_every_epoch = at_every_epoch

    def __call__(self, sess, epoch, iteration, model, loss):
        if iteration == 0 and epoch % self.at_every_epoch == 0:
            total = 0
            correct = 0
            for values in self.batcher:
                total += len(values[-1])
                feed_dict = {}
                for i in range(0, len(self.placeholders)):
                    feed_dict[self.placeholders[i]] = values[i]
                truth = np.argmax(values[-1], 1)
                predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1),
                                     feed_dict=feed_dict)
                correct += sum(truth == predicted)
            acc = float(correct) / total
            self.update_summary(sess, iteration, ACCURACY_TRACE_TAG, acc)
            print("Epoch " + str(epoch) +
                  "\tAcc " + str(acc) +
                  "\tCorrect " + str(correct) + "\tTotal " + str(total))



class BatchBucketSampler:
    """
        Samples batches from a list of data points

        >>> np.random.seed(0)
        >>> seq1s = np.random.choice(3, [2, 3, 1]) #num_ex x #max_seq_length x #input_dim
        >>> seq2s = np.random.choice(5, [2, 3, 2]) #num_ex x #max_seq_length x #input_dim
        >>> targets = np.random.choice(2, [2])
        >>> data = [seq1s, seq2s, targets]
        >>> data
        [array([[[0],
                [1],
                [0]],
        <BLANKLINE>
               [[1],
                [1],
                [2]]]), array([[[4, 0],
                [0, 4],
                [2, 1]],
        <BLANKLINE>
               [[0, 1],
                [1, 0],
                [1, 4]]]), array([1, 0])]
        >>> sampler = BatchBucketSampler(data)
        >>> sampler.get_batch(1)
        [array([[[0],
                [1],
                [0]]]), array([[[4, 0],
                [0, 4],
                [2, 1]]]), array([1])]
        >>> sampler.get_batch(2) # reshuffling takes place
        [array([[[1],
                [1],
                [2]],
        <BLANKLINE>
               [[0],
                [1],
                [0]]]), array([[[0, 1],
                [1, 0],
                [1, 4]],
        <BLANKLINE>
               [[4, 0],
                [0, 4],
                [2, 1]]]), array([0, 1])]
    """
    # todo: add bucketing capabilities by assigning data examples to buckets
    def __init__(self, data, batch_size=1, buckets=None):
        """
        :param data: a list of higher order tensors where the first dimension
        corresponds to the number of examples which needs to be the same for
        all tensors
        :param batch_size: desired batch size
        :param buckets: a list of bucket boundaries
        :return:
        """
        self.data = data
        self.num_examples = len(self.data[0])
        self.batch_size = batch_size
        self.buckets = buckets
        self.to_sample = list(range(0, self.num_examples))
        np.random.shuffle(self.to_sample)
        self.counter = 0

    def __reset(self):
        self.to_sample = list(range(0, self.num_examples))
        np.random.shuffle(self.to_sample)
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_examples - self.counter <= self.batch_size:
            self.__reset()
            raise StopIteration
        return self.get_batch(self.batch_size)

    def get_batch(self, batch_size):
        if self.num_examples == self.counter:
            self.__reset()
            return self.get_batch(batch_size)
        else:
            num_to_sample = batch_size
            batch_indices = []
            if len(self.to_sample) < num_to_sample:
                batch_indices += self.to_sample
                num_to_sample -= len(self.to_sample)
                self.__reset()
            self.counter += batch_size
            batch_indices += self.to_sample[0:num_to_sample]
            self.to_sample = self.to_sample[num_to_sample:]
            return [x[batch_indices] for x in self.data]




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
            print(classification_report(truth_all, pred_all, target_names=["NONE", "AGAINST", "FAVOR"], digits=4))


class AccuracyHookIgnoreNeutral(TraceHook):
    """
    Print accuracy on AGAINST and FAVOR instances only
    """
    def __init__(self, summary_writer, batcher, placeholders, at_every_epoch):
        super().__init__(summary_writer)
        self.batcher = batcher
        self.placeholders = placeholders
        self.at_every_epoch = at_every_epoch

    def __call__(self, sess, epoch, iteration, model, loss):
        if iteration == 0 and epoch % self.at_every_epoch == 0:
            total = 0
            total_old = 0
            correct_old = 0
            correct = 0
            for values in self.batcher:
                total_old += len(values[-1])
                feed_dict = {}
                for i in range(0, len(self.placeholders)):
                    feed_dict[self.placeholders[i]] = values[i]
                truth = np.argmax(values[-1], 1)

                # mask truth
                truth_noneutral = ma.masked_values(truth, 0)
                truth_noneutral_compr = truth_noneutral.compressed()

                predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1),
                                     feed_dict=feed_dict)

                pred_nonneutral = ma.array(predicted, mask=truth_noneutral.mask)
                pred_nonneutral_compr = pred_nonneutral.compressed()

                correct_old += sum(truth == predicted)
                correct += sum(truth_noneutral_compr == pred_nonneutral_compr)
                total += len(truth_noneutral_compr)

            acc = float(correct) / total
            self.update_summary(sess, iteration, "AccurayNonNeut", acc)
            print("Epoch " + str(epoch) +
                  "\tAccNonNeut " + str(acc) +
                  "\tCorrect " + str(correct) + "\tTotal " + str(total))
            return acc
        return 0.0


class SaveModelHookDev(Hook):
    def __init__(self, path, at_every_epoch=5):
        self.path = path
        self.at_every_epoch = at_every_epoch
        self.saver = tf.train.Saver(tf.trainable_variables())

    def __call__(self, sess, epoch, iteration, model, loss):
        if epoch%self.at_every_epoch == 0:
            #print("Saving model...")
            SaveModelHookDev.save_model_dev(self.saver, sess, self.path + "_ep" + str(epoch) + "/", "model.tf")

    def save_model_dev(saver, sess, path, modelname):
        if not os.path.exists(path):
            os.makedirs(path)
        saver.save(sess, os.path.join(path, modelname))


class Trainer(object):
    """
    Object representing a TensorFlow trainer.
    """

    def __init__(self, optimizer, max_epochs, hooks):
        self.loss = None
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.hooks = hooks

    def __call__(self, batcher, placeholders, loss, acc_thresh, pretrain, embedd, sep=False, model=None, session=None):
        self.loss = loss
        minimization_op = self.optimizer.minimize(loss)
        close_session_after_training = False
        if session is None:
            session = tf.Session()
            close_session_after_training = True  # no session existed before, we provide a temporary session

        init = tf.initialize_all_variables()

        if (pretrain == "pre" or pretrain == "pre_cont") and sep == False: # hack if we want to use pre-trained embeddings
            vars = tf.all_variables()
            emb_var = vars[0]
            session.run(emb_var.assign(embedd))
        elif (pretrain == "pre" or pretrain == "pre_cont") and sep == True:
            vars = tf.all_variables()
            emb_var = vars[0]
            emb_var2 = vars[1]
            session.run(emb_var.assign(embedd))
            session.run(emb_var2.assign(embedd))

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

                if isinstance(hook, AccuracyHookIgnoreNeutral):
                    acc = hook(session, epoch, 0, model, 0)
                    if acc > acc_thresh:
                        print("Accuracy threshold reached! Stopping training.")
                        if close_session_after_training:
                            session.close()
                        return epoch
                else:
                    hook(session, epoch, 0, model, 0)
            epoch += 1

        if close_session_after_training:
            session.close()

        return self.max_epochs-1


def load_model_dev(sess, path, modelname):
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, os.path.join(path, modelname))
