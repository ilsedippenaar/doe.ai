# Created by Alison Chi on 10/11/2017
import nltk
from nltk import CFG
from nltk.parse.generate import demo_grammar

from stat_parser import Parser


class SentenceParser:
    def __init__(self, sentence):
        self.sentence = sentence

    def getPOSTags(self):
        tokens = nltk.word_tokenize(self.sentence)
        tagTuples = nltk.pos_tag(tokens)

    def parsePCFG(self):
        parser = Parser()
        parsed = parser.parse(self.sentence)
        return parsed



grammar = CFG.fromstring(demo_grammar)
print(grammar)
