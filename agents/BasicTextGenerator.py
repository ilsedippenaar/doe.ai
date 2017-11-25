import tensorflow as tf
import numpy as np
from collections import Counter
from agents.ChatAgent import ChatAgent


class BasicTextGenerator(ChatAgent):

    def __init__(self, rawData, savePath, stateSize=512, wordsToFeed=50, maxVocabSize=5000):
        """
        The TextGenerator class is trained on a sequence of words and can generate
        an arbitrary sequence of words based on the data.
        :param rawData: A list of words (should all be lowercase and without punctuation)
        :param savePath: The location on disk where to save the model after training
        :param stateSize: The size of the LSTM state vector that maintains a "context" for a point in the sequence
        :param wordsToFeed: The number of iterations in the RNN to do across the sequence
        :param maxVocabSize: The maximum number of words which the model can generate
        """
        super().__init__(rawData, savePath)

        # Architecture
        self.stateSize = stateSize
        self.wordsToFeed = wordsToFeed

        # Learning parameters
        self.numEpochs = 100
        self.batchSize = 50
        self.wordsToFeed = 50
        self.learningRate = 0.01
        self.momentum = 0.5

        self.data, self.wordDict, self.reverseWordDict, self.vocabSize = self._loadData(rawData, maxVocabSize)

    @staticmethod
    def _loadData(rawData, maxVocabSize):
        count = [['UNK', -1]]  # 'UNK' = uncommon token
        count.extend(Counter(rawData).most_common(maxVocabSize - 1))
        wordDict = {wordCount[0]: i for i, wordCount in enumerate(count)}
        data = [wordDict[word] if word in wordDict else 0 for word in rawData]  # unknown words -> 0
        # build reverse dictionary (0 -> 'UNK')
        return data, wordDict, dict(zip(wordDict.values(), wordDict.keys())), len(wordDict)

    def _makeBatch(self):
        # TODO(ilse): make inputs and labels numpy matrices and not lists
        # TODO(ilse): make better randomization method
        inputs, labels = [], []
        for i in range(self.batchSize):
            index = np.random.randint(len(self.data) - self.wordsToFeed - 2)
            # this is necessary to add an extra dimension for the input (right now, each word = 1 int, but eventually
            # one word = n floats)
            inputs.append([[x] for x in self.data[index:(index+self.wordsToFeed)]])
            labels.append(self.data[index+self.wordsToFeed+1])
        return inputs, labels

    def _printSummary(self, step, loss, inputs, labels, predictedLabels):
        print('Loss at step {} = {:.2f}'.format(step, loss))
        if step % (self.numEpochs // 10) == 0:
            print('\tAccuracy = {:.2f} %'.format(100*np.mean(np.array(labels) == np.array(predictedLabels))))
            for batchNum,data in enumerate(inputs):
                text = ' '.join([self.reverseWordDict[x[0]] for x in data])
                print('\tText = {}'.format(text))
                print('\tActual = {}'.format(self.reverseWordDict[labels[batchNum]]))
                print('\tPredicted = {}'.format(self.reverseWordDict[predictedLabels[batchNum]]))

    def _buildModel(self):
        graph = tf.Graph()
        with graph.as_default(): # defines a name scope for all variables below (not necessary)
            # the last 1 in the dimension here could be changed later for word embeddings
            inputs = tf.placeholder(shape=(self.batchSize, self.wordsToFeed, 1), dtype=tf.float32, name='inputs')
            labels = tf.placeholder(shape=(self.batchSize,), dtype=tf.int32, name='labels')
            # TODO(ilse): make this not always constant so that we can feed variable length sentences to the model
            sequence_length = tf.constant([self.wordsToFeed], shape=(self.batchSize,), dtype=tf.int32)

            cell = tf.nn.rnn_cell.BasicLSTMCell(self.stateSize)
            # TODO(ilse): add initialize state for the RNN (i.e. truncated_normal?)
            # dynamic_rnn requires inputs to be 3-dimensional
            output, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=sequence_length, dtype=tf.float32)
            weights = tf.Variable(tf.truncated_normal(shape=(self.batchSize, self.stateSize, self.vocabSize)))
            bias = tf.Variable(tf.zeros(shape=(self.batchSize, self.vocabSize)))

            # get the batchSize x stateSize matrix for the last time point, then add a dimension at pos 1
            output = tf.expand_dims(output[:, -1, :], 1)
            logits = tf.squeeze(tf.matmul(output, weights)) + bias
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            predictedLabels = tf.argmax(logits, 1)
            return graph, inputs, labels, predictedLabels, loss, \
                tf.train.RMSPropOptimizer(learning_rate=self.learningRate, momentum=self.momentum).minimize(loss)

    def testBatch(self, num=10):
        for i in range(num):
            inputs, labels = self._makeBatch()
            print('Input: {}'.format('\n'.join([str(a) for a in inputs])))
            print('Labels: {}'.format(', '.join([str(a) for a in labels])))
            print('\n')

    def train(self):
        graph, inputs, labels, predictedLabels, loss, optimizer = self._buildModel()
        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            for i in range(self.numEpochs):
                batchInputs, batchLabels = self._makeBatch()
                feedDict = {inputs: batchInputs, labels: batchLabels}
                _, predictedLoss, batchPredictedLabels = sess.run([optimizer, loss, predictedLabels], feed_dict=feedDict)
                self._printSummary(i, predictedLoss, batchInputs, batchLabels, batchPredictedLabels)
            saver.save(sess, str(self._savePath))

    def respond(self, prompt):
        if not self._savePath.exists():
            print('Saved model not found at {}'.format(str(self._savePath)))
            return

        tf.reset_default_graph()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, str(self._savePath))
