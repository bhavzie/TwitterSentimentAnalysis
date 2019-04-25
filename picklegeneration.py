'''
	classifying text
	This file creates pickles so that they can be used for any further sentiment analysis
'''

import nltk
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import pandas as pd
import numpy as np
import csv
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
#from collections import Counter

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

#Dataset

#First dataset :
#Documents = List of Dictionaries
#[
	#{ Line1 , pos } , { Line2, neg } ...

#]


short_pos = open('Database/positive', 'r').read()
short_neg = open('Database/negative', 'r').read()


stopWords = set(stopwords.words('english'))

documents = []



all_words = [] # List of all words in entire dataset, most common 3000 words become features

allowed_word_pos = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS" ] 

for r in short_pos.split('\n'):
	documents.append( (r, "pos") )
	words = word_tokenize(r)
	for w in words:
		tagged = nltk.pos_tag([w])
		if (w not in stopWords and w.isalpha() and tagged[0][1] in allowed_word_pos):
			all_words.append(w.lower())

for r in short_neg.split('\n'):
	documents.append( (r, "neg") )
	words = word_tokenize(r)
	for w in words:
		tagged = nltk.pos_tag([w])
		if (w not in stopWords and w.isalpha() and tagged[0][1] in allowed_word_pos):
			all_words.append(w.lower())



#Second Dataset

'''
def translator(user_string):
		user_string  = user_string.split(" ")
		j = 0
		for _str in user_string:
			fileName = "Database/slang"
			accessMode = "r"
			
			with open(fileName, accessMode) as myCsvFile:
				dataFromFile = csv.reader(myCsvFile, delimiter = "=")
				_str = re.sub('[^a-zA-Z0-9-_.]', '', _str)
				for row in dataFromFile:
					if _str.upper() == row[0]:
						user_string[j] = row[1]
				myCsvFile.close()
			j = j + 1
		return(' '.join(user_string))
		
	
def clean_tweet(tweet):
		cleaned_tweet = translator(tweet)
		tok = WordPunctTokenizer()
		pat1 = r'@[A-Za-z0-9]+'
		pat2 = r'https?://[A-Za-z0-9./]+'
		combined_pat = r'|'.join((pat1,pat2))
		www_pat = r'www.[^ ]+'
		negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
				"haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
				"wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
				"can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
				"mustn't":"must not"}
		
		neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
		
		
		soup = BeautifulSoup(cleaned_tweet, 'lxml')
		souped = soup.get_text()
		cleaned_tweet = re.sub(combined_pat, '', souped)
		cleaned_tweet = re.sub(www_pat, '', cleaned_tweet)
		lower_case = cleaned_tweet.lower()
		
		neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
		
		letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
		
		words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
		return (" ".join(words)).strip()
'''

'''
stopWords = set(stopwords.words("english"))
documents = []
all_words = []

with open("Database/train.csv", mode = "r") as csv_file:
	
	csv_reader = csv.DictReader(csv_file)
	
	for dct in map(dict,csv_reader):
		
		inputtext = dct.get("text")
		inputsentiment = dct.get("sentiment")
		
		
		if inputsentiment == "4":
			documents.append( (inputtext, "pos") )
		elif inputsentiment == "0":
			documents.append( (inputtext, "neg") )
		
		
		words = word_tokenize(inputtext)
		for w in words:
			if (w not in stopWords and w.isalpha()):
				all_words.append(w.lower())

'''


save_documents = open("myPickle/documents.pickle", "wb")
pickle.dump(documents,save_documents)
save_documents.close()



print(len(all_words))
all_words = nltk.FreqDist(all_words)
print(len(all_words))

word_features = list(all_words.keys())[:]
print(word_features[:10])

save_word_features = open("myPickle/word_features.pickle" , "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()



def find_features(document):
	words = word_tokenize(document)
	features = { }
	for w in word_features:
		features[w] = (w in words)
		
	return features

featuresets = [ (find_features(rev),category) for (rev,category) in documents ]



save_featuresets = open("myPickle/featuresets.pickle", "wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)

print(len(featuresets))

training_set = featuresets[:8000] 
testing_set = featuresets[8000:]


'''
classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Original Naive Bayes Algo accuracy percent :", (nltk.classify.accuracy(classifier,testing_set))*100 )

classifier.show_most_informative_features(15)

save_classifier = open("myPickle/naivebayes.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()
'''

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier Algo accuracy percent :", (nltk.classify.accuracy(MNB_classifier,testing_set))*100 )

save_classifier = open("myPickle/mnbclassifier.pickle","wb")
pickle.dump(MNB_classifier,save_classifier)
save_classifier.close()


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier Algo accuracy percent :", (nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100 )

save_classifier = open("myPickle/bernoulliNB.pickle","wb")
pickle.dump(BernoulliNB_classifier,save_classifier)
save_classifier.close()


LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Algo accuracy percent :", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100 )

save_classifier = open("myPickle/logisticRegression.pickle","wb")
pickle.dump(LogisticRegression_classifier,save_classifier)
save_classifier.close()


SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier Algo accuracy percent :", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100 )

save_classifier = open("myPickle/Sgdc.pickle","wb")
pickle.dump(SGDClassifier_classifier,save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier Algo accuracy percent :", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100 )

save_classifier = open("myPickle/linearsvc.pickle","wb")
pickle.dump(LinearSVC_classifier,save_classifier)
save_classifier.close()


voted_classifier = VoteClassifier(MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier)

print("Voted_classifier Algo accuracy percent :", (nltk.classify.accuracy(voted_classifier,testing_set))*100 )


def sentiment(text):
	feats = find_features(text)
	
	return voted_classifier.classify(feats)





