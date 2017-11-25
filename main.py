from pathlib import Path
from utils import tokenize, Word2Vec
import tensorflow as tf

from DataSource import DataSource

if __name__ == '__main__':
    rootModelDir = Path('modelData')
    if not rootModelDir.exists():
        rootModelDir.mkdir()
    model1Dir = rootModelDir / 'test_model1'
    if not model1Dir.exists():
        model1Dir.mkdir()

    data = DataSource.ALICE_TEXT.value.getData(force=True)
    # textGenerator = BasicTextGenerator(data, model1Dir, maxVocabSize=1000)
    # textGenerator.testBatch(num=1)
    # textGenerator.train()
    # print(textGenerator.generateText(100))

    data, wordDict, reverseDict = tokenize([x for l in data for x in l], maxVocabSize=5000)
    alice_w2v = Word2Vec(data, rootModelDir / "alice_word2vec", reverseDict=reverseDict, vocabSize=min(len(wordDict), 5000))
    with tf.Session() as sess:
        print(sess.run(alice_w2v.word2vec([wordDict["alice"]])))

