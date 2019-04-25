# TwitterSentimentAnalysis
Fetches tweets from twitter,based on any # given by the user, and gives sentiments along with wordClouds of the most frequent words used in Pos/Neg tweets


Steps To Run :

1) You will need to generate your API keys from https://developer.twitter.com/en/apps
   You can generate the API keys from the above website, then go to twittersentimentanalysis.py and add them on line 31 & 32.
   The code will still work without adding the API keys, because I've already populated the database.csv with some # while testing.
   Although, to get results for a different # you need access to twitter API, for which you need your own keys.

1) Create a Directory named "Database", add the following files to it :
        a) Positive
        b) Negative
        c) Slang
        d) database.csv
        
2) Create a Directory named "myPickle"

3) Create a Directory named "Visual" and add into it :
    a) chart.html
    

Run "picklegeneration.py" and then run "twittersentimentanalysis.py".
