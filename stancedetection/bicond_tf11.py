# This is a small snippet of code illustrating how the bidirectional conditional model can be implemented in Tensorflow 11, with dynamic RNNs

import tensorflow as tf

def create_bi_sequence_embedding(inputs, seq_lengths, repr_dim, vocab_size, emb_name, rnn_scope, reuse_scope=False, _FLOAT_TYPE=tf.float64):
    """
    LSTM encoding - forward and backward reading of first LSTM
    :param inputs: tensor [d1, ... ,dn] of int32 symbols
    :param seq_lengths: [s1, ..., sn] lengths of instances in the batch
    :param repr_dim: dimension of embeddings
    :param vocab_size: number of symbols
    :param emb_name: name of embedding matrix
    :param rnn_scope: name of RNN scope
    :param reuse_scope: reuse the RNN scope or not
    :return: return outputs_fw, last_state_fw, outputs_bw, last_state_bw
    """

    # use a shared embedding matrix for now, test if this outperforms separate matrices later
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, repr_dim], -0.1, 0.1, dtype=_FLOAT_TYPE),
                                   name=emb_name, trainable=True, dtype=_FLOAT_TYPE)
    # [batch_size, max_seq_length, input_size]
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)

    # dummy test to see if the embedding lookup is working
    # Reduce along dimension 1 (`n_input`) to get a single vector (row) per input example
    # embedding_aggregated = tf.reduce_sum(embedded_inputs, [1])


    ### first FW LSTM ###
    with tf.variable_scope(rnn_scope + "_FW") as scope:
        if reuse_scope == True:
            scope.reuse_variables()
        cell_fw = tf.nn.rnn_cell.LSTMCell(repr_dim, state_is_tuple=True)
        #cell_fw = tf.contrib.rnn.AttentionCellWrapper(cell_fw, 3, state_is_tuple=True) # not working
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_fw, output_keep_prob=0.9)
        # outputs shape: [batch_size, max_time, cell.output_size]
        # last_states shape: [batch_size, cell.state_size]
        outputs_fw, last_state_fw = tf.nn.dynamic_rnn(
            cell=cell_fw,
            dtype=_FLOAT_TYPE,
            sequence_length=seq_lengths,
            inputs=embedded_inputs)


    embedded_inputs_rev = tf.reverse(embedded_inputs, [False, True, False])  # reverse the sequence

    ### first BW LSTM ###
    with tf.variable_scope(rnn_scope + "_BW") as scope:
        if reuse_scope == True:
            scope.reuse_variables()
        cell_bw = tf.nn.rnn_cell.LSTMCell(repr_dim, state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_bw, output_keep_prob=0.9)

        # outputs shape: [batch_size, max_time, cell.output_size]
        # last_states shape: [batch_size, cell.state_size]
        outputs_bw, last_state_bw = tf.nn.dynamic_rnn(
            cell=cell_bw,
            dtype=_FLOAT_TYPE,
            sequence_length=seq_lengths,
            inputs=embedded_inputs_rev)


    # return outputs of LSTMs, to be fed into create_bi_sequence_embedding_initialise()
    return outputs_fw, last_state_fw, outputs_bw, last_state_bw, embedding_matrix


def create_bi_sequence_embedding_initialise(inputs_cond, seq_lengths_cond, repr_dim, rnn_scope_cond, last_state_fw, last_state_bw, embedding_matrix, reuse_scope=False, _FLOAT_TYPE=tf.float64):
    """
    Bidirectional conditional encoding
    :param repr_dim: dimension of embeddings
    :param vocab_size: number of symbols
    :param emb_name: name of embedding matrix
    :param rnn_scope: name of RNN scope
    :param reuse_scope: reuse the RNN scope or not
    :return: return [batch_size, repr_dim] tensor representation of symbols.
    """

    ### second FW LSTM ###

    embedded_inputs_cond = tf.nn.embedding_lookup(embedding_matrix, inputs_cond) # [batch_size, max_seq_length, input_size]

    # initialise with state of context
    with tf.variable_scope(rnn_scope_cond + "_FW") as scope:
        if reuse_scope == True:
            scope.reuse_variables()
        cell_fw_cond = tf.nn.rnn_cell.LSTMCell(repr_dim, state_is_tuple=True)
        cell_fw_cond = tf.nn.rnn_cell.DropoutWrapper(cell=cell_fw_cond, output_keep_prob=0.9)

        # returning [batch_size, max_time, cell.output_size]
        outputs_fw_cond, last_state_fw_cond = tf.nn.dynamic_rnn(
            cell=cell_fw_cond,
            dtype=_FLOAT_TYPE,
            sequence_length=seq_lengths_cond,
            inputs=embedded_inputs_cond,
            initial_state=last_state_fw
        )

    embedded_inputs_cond_rev = tf.reverse(embedded_inputs_cond, [False, True, False])  # reverse the sequence


    ### second BW LSTM ###

    with tf.variable_scope(rnn_scope_cond + "_BW") as scope:
        if reuse_scope == True:
            scope.reuse_variables()
        cell_fw = tf.nn.rnn_cell.LSTMCell(repr_dim, state_is_tuple=True)
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_fw, output_keep_prob=0.9)

        # outputs shape: [batch_size, max_time, cell.output_size]
        # last_states shape: [batch_size, cell.state_size]
        outputs_bw_cond, last_state_bw_cond = tf.nn.dynamic_rnn(
            cell=cell_fw,
            dtype=_FLOAT_TYPE,
            sequence_length=seq_lengths_cond,
            inputs=embedded_inputs_cond_rev,
            initial_state=last_state_bw
        )


    # version 1 for getting last output
    #last_output_fw = tfutil.get_by_index(outputs_fw_cond, seq_lengths_cond)
    #last_output_bw = tfutil.get_by_index(outputs_bw_cond, seq_lengths_cond)

    # version 2 for getting last output, without slicing, see http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
    # input, seq_lengths, seq_dim, batch_dim=None, name=None
    # might be more efficient or not, but at least memory warning disappears
    outputs_fw = tf.reverse_sequence(outputs_fw_cond, seq_lengths_cond, seq_dim=1, batch_dim=0)  # slices of input are reversed on seq_dim, but only up to seq_lengths
    dim1fw, dim2fw, dim3fw = tf.unpack(tf.shape(outputs_fw)) #[batch_size, max_time, cell.output_size]
    last_output_fw = tf.reshape(tf.slice(outputs_fw, [0, 0, 0], [dim1fw, 1, dim3fw]), [dim1fw, dim3fw])

    outputs_bw = tf.reverse_sequence(outputs_bw_cond, seq_lengths_cond, seq_dim=1, batch_dim=0)  # slices of input are reversed on seq_dim, but only up to seq_lengths
    dim1bw, dim2bw, dim3bw = tf.unpack(tf.shape(outputs_bw)) #[batch_size, max_time, cell.output_size]
    last_output_bw = tf.reshape(tf.slice(outputs_bw, [0, 0, 0], [dim1bw, 1, dim3bw]), [dim1bw, dim3bw])



    outputs_fin = tf.concat(1, [last_output_fw, last_output_bw])

    #print(tf.shape(last_output_bw))
    #print(tf.shape(outputs_fin))


    return outputs_fin



def create_bicond_embeddings_reader(first_seq, first_seq_lens, second_seq, second_seq_lens, dim, num_symbols):
    """
    Create a bidirectional conditional LSTM reader, using the two helper functions create_bi_sequence_embedding() and create_bi_sequence_embedding_initialise()
    :param first_seq first: sequence, dimensionality: [batch_size, num_tokens]
    :param first_seq_lens: sequence lengths, dimensionality: [batch_size]
    :param second_seq: sequence, dimensionality: [batch_size, num_tokens]
    :param second_seq_lens: sequence lengths, dimensionality: [batch_size]
    :param dim: dimensionality
    :param num_symbols: number of vocab symbols
    """


    # 1) run first LSTM to encode the first sequence
    outputs_fw, last_state_fw, outputs_bw, last_state_bw, embedding_matrix = create_bi_sequence_embedding(first_seq, first_seq_lens, dim,
                                                         num_symbols,
                                                         "embedding_matrix", "RNN_c", reuse_scope=False)

    # 2) run second LSTM to encode the second sequence, taking output of first sequence as input
    seq_pair_encoding = create_bi_sequence_embedding_initialise(second_seq, second_seq_lens, dim, "RNN_q", last_state_fw,
                                                                        last_state_bw, embedding_matrix, reuse_scope=True)

    return seq_pair_encoding
