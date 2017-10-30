from DataSource import DataSource
from TextGenerator import TextGenerator
from pathlib import Path

if __name__ == '__main__':
    rootModelDir = Path('modelData')
    if not rootModelDir.exists():
        rootModelDir.mkdir()
    model1Dir = rootModelDir / 'test_model1'
    if not model1Dir.exists():
        model1Dir.mkdir()

    data = DataSource.ALICE_TEXT.value.getData()
    textGenerator = TextGenerator(data, model1Dir, maxVocabSize=1000)
    #textGenerator.testBatch(num=1)
    textGenerator.train()
    #print(textGenerator.generateText(100))

