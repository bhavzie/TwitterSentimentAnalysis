'''
	classifying text
	This file creates uses created pickles to perform the sentiment analysis
'''

import nltk
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

from warnings import simplefilter
simplefilter(action = "ignore", category = FutureWarning)

class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers
	
	def classify(self, features):
		# Returns the most common label returned by each classifier
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)
	
	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		
		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf



load_documents = open("myPickle/documents.pickle", "rb")
documents = pickle.load(load_documents)
load_documents.close()


load_word_features = open("myPickle/word_features.pickle", "rb")
word_features = pickle.load(load_word_features)
load_word_features.close()


def find_features(document):
	words = word_tokenize(document)
	features = { }
	for w in word_features:
		features[w] = (w in words)
		
	return features

load_featuresets = open("myPickle/featuresets.pickle", "rb")
featuresets = pickle.load(load_featuresets)
load_featuresets.close()

load_classifier = open("myPickle/mnbclassifier.pickle", "rb")
MNB_classifier = pickle.load(load_classifier)
load_classifier.close()

load_classifier = open("myPickle/bernoulliNB.pickle", "rb")
BernoulliNB_classifier = pickle.load(load_classifier)
load_classifier.close()

load_classifier = open("myPickle/logisticRegression.pickle", "rb")
LogisticRegression_classifier = pickle.load(load_classifier)
load_classifier.close()

load_classifier = open("myPickle/Sgdc.pickle", "rb")
SGDClassifier_classifier = pickle.load(load_classifier)
load_classifier.close()

load_classifier = open("myPickle/linearsvc.pickle", "rb")
LinearSVC_classifier = pickle.load(load_classifier)
load_classifier.close()

'''
load_classifier = open("myPickle/Nusvc.pickle", "rb")
NuSVC_classifier = pickle.load(load_classifier)
load_classifier.close()
'''


voted_classifier = VoteClassifier(MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier)

def sentiment(text):
	feats = find_features(text)
	
	return voted_classifier.classify(feats), voted_classifier.confidence(feats)




