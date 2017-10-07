import numpy as np
import tensorflow as tf
from collections import Counter


class TextGenerator:

    def __init__(self, dataFile, stateSize=512, numHidden=50, vocabSize=50000):
        self.dataFile = dataFile

        # Architecture
        self.stateSize = stateSize
        self.numHidden = numHidden
        self.vocabSize = vocabSize

        # Learning parameters
        self.numEpochs = 1000
        self.batchSize = 100
        self.wordsToFeed = 50
        self.learningRate = 0.01
        self.momentum = 0.5

        self.data, self.wordDict, self.reverseWordDict = self._loadData()

    def _loadData(self):
        words = []
        with open(self.dataFile, 'r') as f:
            words.append(f.readline().split())
        count = [['UNK', -1]]
        count.extend(Counter(words).most_common(self.vocabSize - 1))
        wordDict = {wordCount[0]: i for i, wordCount in enumerate(count)}
        data = [wordDict[word] if word in wordDict else 0 for word in words]
        return data, wordDict, dict(zip(wordDict.values(), wordDict.keys()))

    def _makeBatch(self):
        inputs, labels = [], []
        for i in range(self.batchSize):
            index = np.random.randint(len(self.data) - self.wordsToFeed - 2)
            inputs.append(self.data[index:(index+self.wordsToFeed)])
            labels.append(self.data[index+self.wordsToFeed+1])
        return inputs, labels

    def _printSummary(self, step, loss, labels, predictedLabels):
        print('Loss at step {} = {:5f}'.format(step, loss))
        if step % self.numEpochs // 10 == 0:
            print('\tAccuracy = {:5f}'.format(np.mean(np.array(labels) == np.array(predictedLabels))))

    def _buildModel(self):
        graph = tf.Graph()
        with graph.as_default():
            inputs = tf.placeholder(dtype=tf.int64, shape=(self.batchSize, self.wordsToFeed, self.vocabSize))
            labels = tf.sparse_placeholder(dtype=tf.int64, shape=(self.batchSize, self.vocabSize))

            cell = tf.contrib.rnn.BasicLSTMCell(self.stateSize)
            output, _ = tf.nn.dynamic_rnn(cell, inputs)
            weights = tf.truncated_normal(shape=(self.batchSize, self.stateSize, self.vocabSize))
            bias = tf.truncated_normal(shape=(self.batchSize, self.vocabSize))

            logits = tf.matmul(output, weights) + bias
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return graph, inputs, labels, loss

    def train(self, save=True):
        graph, inputs, labels, loss = self._buildModel()
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learningRate, momentum=self.momentum).minimize(loss)
        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()
            for i in range(self.numEpochs):
                batchInputs, batchLabels = self._makeBatch()
                feedDict = {inputs: batchInputs, labels: batchLabels}
                _, predictedLoss, predictedLabels = sess.run([optimizer, loss, labels], feed_dict=feedDict)
                self._printSummary(i, predictedLoss, batchLabels, predictedLabels)

    def generateText(self, numWords):
        pass
