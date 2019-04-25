'''
get data in json format		// ml.py
this could either be streamed(realtime) or api.search(past 7 days)
store in file			// ml.py
get tweet.text from json	// sdt.py
now sentiment using textblob	// sdt.py
'''

import tweepy
from tweepy import Stream
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
import re
import csv
from textblob import TextBlob
import sentimentanalysis as snltk
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from collections import Counter
import vincent
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class TwitterClient(object):
	
	databasefile = "Database/database.csv"
	def __init__(self):
		consumer_key = "aMnpUWTs3fctzlGKBOjn551HX"
		consumer_secret = "rDnh0B9qYUo2iQE2jveRAga4Ap9NdHEpW1QAyIoSkE3M0uOAxf"

		access_token = "2447293766-t0YpormwzErih4GTZuTBuafu2U4VIY6J6rQReqq"
		access_token_secret = "wOalP9hxeaUu5wqKdTDmj9CK66IjItvstSlvLGNKrDvJ3"

		try:
			self.auth = OAuthHandler(consumer_key,consumer_secret)
			self.auth.set_access_token(access_token,access_token_secret)
			self.api = tweepy.API(self.auth)
		except:
			print("Error : Authentication Failed")
	
	def translator(self, user_string):
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
		
	
	def clean_tweet(self,tweet):
			cleaned_tweet = self.translator(tweet)
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
		
		
	def get_tweets(self,query,count = 10):
		tweets = []
		
		try:
			fetched_tweets = self.api.search(q = query, count = count, lang = "en")
			
			'''
				Once we got our tweets in Json format we will write them
				to a file creating a tweet database to work in the future
				Now I'll try to choose what attributes of a tweet object 
				will be needed for our sentiment analysis :
				Consider a tweet object :  
				{
					created_at (The more recent can be given more value)
					text (Actual Tweet)
					place (If i want to limit my search to given region)
					favorite_count (No of likes)
					retweet_count (More retweets more important)
					lang (If i want to filter by language)
				}
				
			'''
			
			for tweet in fetched_tweets:
				parsed_tweet = {}
				
				parsed_tweet['created_at'] = tweet.created_at
				
				tweetcleaned = self.clean_tweet(tweet.text)
				tweetfinaltext = ""
				stopWords = set(stopwords.words('english'))
				
				tweetcleaned = word_tokenize(tweetcleaned)
				
				tweetcleaned = [w for w in tweetcleaned if w not in stopWords] 
				
				for w in tweetcleaned:
					tweetfinaltext = tweetfinaltext + " " + w
				
				parsed_tweet['text'] = tweetfinaltext
				parsed_tweet['place'] = tweet.place
				parsed_tweet['favorite_count'] = tweet.favorite_count
				parsed_tweet['retweet_count'] = tweet.retweet_count
				parsed_tweet['lang'] = tweet.lang
				
				if tweet.retweet_count > 0:
					if parsed_tweet not in tweets:
						tweets.append(parsed_tweet)
				else:
					tweets.append(parsed_tweet)
			'''
				At this point "tweets" must include all our tweets in our own
				Json-style format : 
				{
					created_at : "	"
					text : " "
					place : " "
					favorite_count : " "
					retweet_count : " "
					lang : " "
				}
				
			'''
			fields = ['created_at', 'text', 'place', 'favorite_count', 'retweet_count', 'lang']
			
			with open(self.databasefile, 'w') as csvfile:
				writer = csv.DictWriter(csvfile,fieldnames = fields)
				writer.writeheader()
				writer.writerows(tweets)
		
			
		except tweepy.TweepError as e:
			print("Error : "+str(e))
			
	def get_basic_sentiment(self):
		
		tweettextanalysis = []
		
		#Reading our db file
		with open(self.databasefile, 'r') as csvfile:
			reader = csv.DictReader(csvfile)
			
			#Carefully only collecting our tweet.text for the basic sentiment analysis
			for dct in map(dict,reader):
				parsed_tweet = {}
				
				parsed_tweet['text'] = dct.get("text")
				parsed_tweet['sentiment'] = self.textblob_analysis(dct.get("text"))
				
				tweettextanalysis.append(parsed_tweet)
		
		return tweettextanalysis		
				
	
	def textblob_analysis(self,tweet):
		
		analysis = TextBlob(self.clean_tweet(tweet))
		
		if analysis.sentiment.polarity > 0:
			return 'positive'
		elif analysis.sentiment.polarity == 0:
			return 'neutral'
		else:
			return 'negative'
	
	def main_analysis(self):
		tweettextanalysis = []
		
		#Reading our db file
		with open(self.databasefile, 'r') as csvfile:
			reader = csv.DictReader(csvfile)
			
			#Carefully only collecting our tweet.text for the sentiment analysis
			for dct in map(dict,reader):
				parsed_tweet = {}
				
				parsed_tweet['text'] = dct.get("text")
				parsed_tweet['sentiment'] = snltk.sentiment(dct.get("text"))
				
				tweettextanalysis.append(parsed_tweet)
		
		return tweettextanalysis		
		
	
	def displaymost_commonwords(self):
		all_words = []
		
		
		with open(self.databasefile, 'r') as csvfile:
			reader = csv.DictReader(csvfile)
			
			for dct in map(dict,reader):
				words = word_tokenize(dct.get('text'))
				for w in words:
					all_words.append(w)
		
		word_freq = Counter(all_words).most_common(10)
		labels, freq = zip(*word_freq)
		
		data = {'data' : freq, 'x':labels}
		
		bar = vincent.Bar(data,iter_idx = 'x')
		bar.to_json('Visual/term_freq.json')
		
		text = ""
		for word in all_words:
			text = text + ' ' + word
		
		wordcloud = WordCloud(width=480, height=480, max_words=100).generate(text) 
  
		# plot the WordCloud image  
		plt.figure() 
		plt.imshow(wordcloud, interpolation="bilinear") 
		plt.suptitle('Most Common words in all tweets')
		plt.axis("off") 
		plt.margins(x=0, y=0) 
		plt.show() 
	
	
	def displaymost_commonwordstype(self, typetweets, flag):
		
		text1 = ""
		
		
		for tweet in typetweets:
			text1 = text1 + ' ' + tweet['text']
		
		wordcloud = WordCloud(width=480, height=480, max_words=100).generate(text1) 
  
		# plot the WordCloud image  
		plt.figure() 
		plt.imshow(wordcloud, interpolation="bilinear")
		
		if flag == 1:
			plt.suptitle('Most common words in negative tweets')
		else:
			plt.suptitle('Most common words in positive tweets')
		plt.axis("off") 
		plt.margins(x=0, y=0) 
		plt.show() 
	
	
def main():
	print("entry : ")

	val = input("Enter the tag you want to search on twitter :	")
	
	api = TwitterClient()
	api.get_tweets(query = val, count = 100)
	
	# This is the most basic analysis, gives % positive and % negative tweets
	textblob_analysis_results = api.get_basic_sentiment()
	
	
	ptweets = [tweet for tweet in textblob_analysis_results if tweet['sentiment'] == 'positive']
	
	ntweets = [tweet for tweet in textblob_analysis_results if tweet['sentiment'] == 'negative']
	
	print("Based on basic Sentiment analysis :	")
	print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(textblob_analysis_results)))
	
	print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(textblob_analysis_results)))
	
	api.displaymost_commonwordstype(ptweets,0)
	api.displaymost_commonwordstype(ntweets,1)
	
	'''
	print("Neutral tweets percentage: {} % \ ".format(100*( len(tweets) - len(ntweets) - len(ptweets) )/len(tweets))) 
	
	print("\n\nPositive tweets:") 
	for tweet in ptweets[:5]: 
		print(tweet['text'])
	
	print("\n\nNegative tweets:") 
	for tweet in ntweets[:5]: 
		print(tweet['text'])
	'''
	
	main_analysis_results = api.main_analysis()
	
	postweets = [tweet for tweet in main_analysis_results if tweet['sentiment'][0] == 'pos']
	
	negtweets = [tweet for tweet in main_analysis_results if tweet['sentiment'][0] == 'neg']
	
	
	print()
	print()
	
	
	print("Based on Main Sentiment analysis :	")
	print("Positive tweets percentage: {} %".format(100*len(postweets)/len(main_analysis_results)))
	
	print("Negative tweets percentage: {} %".format(100*len(negtweets)/len(main_analysis_results)))
	
	api.displaymost_commonwords()
	
	api.displaymost_commonwordstype(postweets,0)
	api.displaymost_commonwordstype(negtweets,1)
	
	#for w in main_analysis_results[:10]:
		#print(w)
	
	
if __name__ == "__main__":
	main()
	


'''
class MyStreamListener(StreamListener):
	
	def on_data(self,data):
		all_data = json.loads(data)
		print(all_data["text"])
		return True
			
	def on_error(self,status_code):
		print(status)
		return True
		
myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth,listener = myStreamListener)

myStream.filter(track = ['#cricket'])

'''
