# Created by Alison Chi on 10/11/2017
import nltk
from nltk import CFG
from nltk.parse.generate import demo_grammar
from nltk.corpus import treebank
from itertools import islice
from nltk.grammar import PCFG, induce_pcfg, toy_pcfg1, toy_pcfg2

from nltk import ParserI
from nltk.parse import ViterbiParser
from nltk.grammar import toy_pcfg1, toy_pcfg2
from nltk import ProbabilisticTree
from nltk import ProbabilisticDependencyGrammar
from nltk import ProbabilisticNonprojectiveParser
from nltk import ProbabilisticProjectiveDependencyParser
from nltk import BottomUpProbabilisticChartParser
from nltk import ProbabilisticProduction
import nltk
from nltk.corpus import treebank
from nltk.grammar import PCFG, Nonterminal
import language_check

# do PCFG demo
# no FRAG, X, NX, same word two in a row,

class SentenceParser:
    def __init__(self):
        self.checker = language_check.LanguageTool('en-US')

    def getPOSTags(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        tagTuples = nltk.pos_tag(tokens)
        return tagTuples

    def isCorrect(self, sentence):
        checked = self.checker.check(sentence.capitalize())
        if len(checked) == 0:
            return True
        return False



#parser = SentenceParser("The cat ate the mouse.")
#tagged = parser.getPOSTags()
#print(tagged)
#print(parser.getPCFG())

# language check stuff
tool = language_check.LanguageTool('en-US')

text1 = "The cat ate the mouse."
text2 = "Cat the are the mouse."
text3 = "I I I I I I."
text4 = "Smompy druid 20 darns."
text5 = "Dog wanted dug."
text6 = "Tell me"

parser = SentenceParser()
print(parser.isCorrect(text1))
print(parser.isCorrect(text2))



