from DataSource import DataSource
from TextGenerator import TextGenerator
from pathlib import Path

if __name__ == '__main__':
    data = DataSource.ALICE_TEXT.value.getData()
    textGenerator = TextGenerator(data, Path('modelData/test_model1.mod'))
    #textGenerator.testBatch()
    textGenerator.train()
    #print(textGenerator.generateText(100))

