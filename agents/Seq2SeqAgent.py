import tensorflow as tf

from agents.ChatAgent import ChatAgent


class Seq2SeqAgent(ChatAgent):
    def __init__(self, data, savePath):
        super().__init__(data, savePath)

        self.embedding_size = 1
        self.stateSize = 500
        self.batchSize = 50

    def _buildModel(self):
        # variable length time
        encoder_inputs = tf.placeholder(tf.float32, shape=(self.batchSize, None, self.embedding_size),
                                name="encoder_inputs")
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.stateSize)
        encoder_outputs, _ = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs, sequence_length=seq_len)

    def train(self):
        super().train()

    def respond(self, prompt):
        return super().respond(prompt)

