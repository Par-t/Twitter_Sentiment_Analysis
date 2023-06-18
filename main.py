import streamlit as st

import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
import tweepy
import re

consumer_key = st.secrets["CONSUMER_KEY"]
bearer = st.secrets["BEARER"]
access_token = st.secrets["ACCESS_TOKEN"]
access_token_secret = st.secrets["ACCESS_TOKEN_SECRET"]
consumer_secret = st.secrets["CONSUMER_SECRET"]

print(consumer_secret)

nltk.download('punkt')
nltk.download('stopwords')

st.title("Real time Twitter data Sentiment Analysis")

client = tweepy.Client(bearer, consumer_key, consumer_secret, access_token, access_token_secret)
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

tag = st.text_input('Enter Hashtag for tweets to be fetched:', '#')
print(tag)

if len(tag) > 1 & st.button('Analyze Tweets'):
    query = tag
    tweets = tweepy.Cursor(api.search_tweets, q=query, tweet_mode='extended').items(limit=500)

    tweet_list = []
    for tweet in tweets:
        tweet_list.append(tweet.full_text)

    print(len(tweet_list))

    from googletrans import Translator

    detector = Translator()
    data = []
    for tweet in tweet_list:
        dec_lan = detector.detect(tweet)

        if (dec_lan.lang == 'en' and dec_lan.confidence >= 0.7):
            data.append(tweet)

    print(len(data))


    def pp(text):

        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
        text = re.sub(r'#', '', text)

        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        text = tokenizer.tokenize(text)

        y = []
        for i in text:
            if i.isalnum():
                y.append(i)

        text = y[:]
        y.clear()

        stop_words = set(stopwords.words('english'))
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for i in text:
            if i not in stop_words and i not in punc:
                y.append(i)

        text = y[:]
        y.clear()

        ps = PorterStemmer()
        for i in text:
            y.append(ps.stem(i))

        return " ".join(y)


    df = []

    df = pd.DataFrame(data)

    df.rename(columns={df.columns[0]: 'text'}, inplace=True)

    df['pp_tweet'] = df['text'].apply(pp)

    tfidf = pickle.load(open('sentiment_analysis_vectorizer.pkl', 'rb'))
    model = pickle.load(open('sentiment_analysis_model.pkl', 'rb'))

    vector_input = tfidf.transform([df['pp_tweet'][0]])

    result = model.predict(vector_input)[0]

    print(result)

    vector_input = tfidf.transform(df['pp_tweet']).toarray()
    pred = model.predict(vector_input)
    result = []
    j = 0
    for i in pred:
        if i == 1:
            result.append("Positive")
        elif i == 0:
            result.append("Negative")

    dfResult = pd.DataFrame()
    dfResult['Tweet'] = df['text']
    dfResult['Sentiment'] = result

    st.write(dfResult)
