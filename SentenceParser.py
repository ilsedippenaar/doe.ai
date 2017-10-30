# Created by Alison Chi on 10/11/2017
import nltk
from nltk import CFG
from nltk.parse.generate import demo_grammar

from nltk import ParserI
from nltk import ProbabilisticTree
from nltk import ProbabilisticDependencyGrammar
from nltk import ProbabilisticNonprojectiveParser
from nltk import ProbabilisticProjectiveDependencyParser
from nltk import BottomUpProbabilisticChartParser
from nltk import ProbabilisticProduction
import nltk
from nltk.corpus import treebank
from nltk.grammar import PCFG, Nonterminal

# do PCFG demo



class SentenceParser:
    def __init__(self, sentence):
        self.sentence = sentence

    def getPOSTags(self):
        tokens = nltk.word_tokenize(self.sentence)
        tagTuples = nltk.pos_tag(tokens)
        return tagTuples

    def getPCFG(self):
        simpleG = "S -> NP VP [1.0] " \
                  "NP -> DT NN [0.3] " \
                  "NP -> NN [0.3] " \
                  "NP -> DT JJ NN [0.3] "  \
                  "NP -> NP PP [0.1] " \
                  "VP -> V NP [0.2] " \
                  "VP -> V [0.3] " \
                  "VP -> VP JJ [0.1] " \
                  "VP -> VP PP [0.1] " \
                  "VP -> VP RB [0.2] " \
                  "VP -> V NP PP [0.1] " \
                  "PP -> PRP NP [1.0]"
        pcfg = nltk.PCFG.fromstring(simpleG)  # add relative and superlative clauses + probabilities
        return pcfg


parser = SentenceParser("The cat ate the mouse.")
tagged = parser.getPOSTags()
print(tagged)
#print(parser.getPCFG())

#tbank_productions = set(production for sent in treebank.parsed_sents() for production in sent.productions())
#tbank_grammar = PCFG(Nonterminal('S'), list(tbank_productions))
mini_grammar = PCFG(Nonterminal('S'), treebank.parsed_sents()[0].productions())
parser = nltk.parse.EarleyChartParser(mini_grammar)
print(parser.parse(treebank.sents()[0]))

