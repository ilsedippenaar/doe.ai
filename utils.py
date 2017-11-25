import tensorflow as tf
import numpy as np
from collections import Counter, deque
from six.moves import xrange
import random
import math


def tokenize(words, maxVocabSize=10000):
    count = [['UNK', -1]]  # 'UNK' = uncommon token, -1 doesn't matter at all
    count.extend(Counter(words).most_common(maxVocabSize - 1))
    wordDict = {wordCount[0]: i for i, wordCount in enumerate(count)}
    data = [wordDict[word] if word in wordDict else 0 for word in words]  # unknown words -> 0
    # build reverse dictionary (0 -> 'UNK')
    return data, wordDict, dict(zip(wordDict.values(), wordDict.keys()))


class Word2Vec:
    """
    Shamelessly stolen from TensorFlow's word2vec tutorial. We don't need anything more complicated at the moment.
    """
    def __init__(self, data, savePath, vocabSize=10000, reverseDict=None):
        self.data = data  # needs to be word ids
        self.savePath = savePath
        self.vocabSize = vocabSize
        self.reverseDict = reverseDict
        if self.savePath.exists():
            with open(savePath, "r") as f:
                size = [int(x) for x in f.readline().split()]
                self.embeddings = np.ndarray(shape=size, dtype=np.float32)
                for i,line in enumerate(f):
                    self.embeddings[i,:] = [float(x) for x in line.split()]
        else:
            self.embeddings = None

        self._dataIndex = 0

        self._batchSize = 128
        self._embeddingSize = 128  # Dimension of the embedding vector.
        self._skipWindow = 1  # How many words to consider left and right.
        self._numSkips = 2  # How many times to reuse an input to generate a label.

        self.valid_size = 16  # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)

    def _generateBatch(self, batchSize, numSkips, skipWindow):
        assert batchSize % numSkips == 0
        assert numSkips <= 2 * skipWindow
        batch = np.ndarray(shape=(batchSize,), dtype=np.int32)
        labels = np.ndarray(shape=(batchSize, 1), dtype=np.int32)
        span = 2 * skipWindow + 1  # [ skip_window target skip_window ]
        buffer = deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self._dataIndex])
            self._dataIndex = (self._dataIndex + 1) % len(self.data)
        for i in range(batchSize // numSkips):
            target = skipWindow  # target label at the center of the buffer
            targets_to_avoid = [skipWindow]
            for j in range(numSkips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * numSkips + j] = buffer[skipWindow]
                labels[i * numSkips + j, 0] = buffer[target]
            buffer.append(self.data[self._dataIndex])
            self._dataIndex = (self._dataIndex + 1) % len(self.data)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self._dataIndex = (self._dataIndex + len(self.data) - span) % len(self.data)
        return batch, labels

    def _buildModel(self):
        num_sampled = 64  # Number of negative examples to sample.

        graph = tf.Graph()
        with graph.as_default():
            # Input data.
            train_inputs = tf.placeholder(tf.int32, shape=[self._batchSize])
            train_labels = tf.placeholder(tf.int32, shape=[self._batchSize, 1])
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                embeddings = tf.Variable(
                    tf.random_uniform([self.vocabSize, self._embeddingSize], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                    tf.truncated_normal([self.vocabSize, self._embeddingSize],
                                        stddev=1.0 / math.sqrt(self._embeddingSize)))
                nce_biases = tf.Variable(tf.zeros([self.vocabSize]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_labels,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=self.vocabSize))

            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = tf.divide(embeddings, norm)
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

            # Add variable initializer.
            init = tf.global_variables_initializer()
        return init, graph, train_inputs, train_labels, normalized_embeddings, similarity, optimizer, loss

    def train(self):
        num_steps = 10001 # 100001
        init, graph, train_inputs, train_labels, normalized_embeddings, similarity, optimizer, loss = self._buildModel()
        with tf.Session(graph=graph) as session:
            # We must initialize all variables before we use them.
            init.run()
            print("Initialized")

            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = self._generateBatch(
                    self._batchSize, self._numSkips, self._skipWindow)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if self.reverseDict and step % 10000 == 0:
                    sim = similarity.eval()
                    for i in xrange(self.valid_size):
                        valid_word = self.reverseDict[self.valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in xrange(top_k):
                            close_word = self.reverseDict[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)
            self.embeddings = normalized_embeddings.eval()

        with open(self.savePath, "w") as f:
            f.write(" ".join([str(x) for x in self.embeddings.shape]) + "\n")
            for wordEmbedding in self.embeddings:
                f.write(" ".join([str(x) for x in wordEmbedding]) + "\n")

    def word2vec(self, wordIds):
        return tf.nn.embedding_lookup(self.embeddings, wordIds)