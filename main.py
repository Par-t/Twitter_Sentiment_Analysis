import streamlit as st

import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
import tweepy
import re
import numpy as np
from googletrans import Translator

consumer_key = st.secrets["CONSUMER_KEY"]
bearer = st.secrets["BEARER"]
access_token = st.secrets["ACCESS_TOKEN"]
access_token_secret = st.secrets["ACCESS_TOKEN_SECRET"]
consumer_secret = st.secrets["CONSUMER_SECRET"]

nltk.download('punkt')
nltk.download('stopwords')

st.title("Real time Twitter data Sentiment Analysis")

tag = st.text_input('Enter Hashtag for tweets to be fetched:', '#')
print(tag)
clicked = st.button('Analyze Tweets')

if len(tag) > 1 and clicked == 1:

    client = tweepy.Client(bearer, consumer_key, consumer_secret, access_token, access_token_secret)
    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
    api = tweepy.API(auth)
    query = tag
    tweets = tweepy.Cursor(api.search_tweets, q=query, tweet_mode='extended').items(limit=100)

    tweet_list = []
    for tweet in tweets:
        tweet_list.append(tweet.full_text)

    print(len(tweet_list))


    detector = Translator()
    data = []
    for tweet in tweet_list:
        dec_lan = detector.detect(tweet)

        if dec_lan.lang == 'en' and dec_lan.confidence >= 0.7:
            data.append(tweet)

    print(len(data))


    def pp(text):

        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
        text = re.sub(r'#', '', text)

        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        text = tokenizer.tokenize(text)

        y = []
        for j in text:
            if j.isalnum():
                y.append(j)

        text = y[:]
        y.clear()

        stop_words = set(stopwords.words('english'))
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for j in text:
            if j not in stop_words and j not in punc:
                y.append(j)

        text = y[:]
        y.clear()

        ps = PorterStemmer()
        for j in text:
            y.append(ps.stem(j))

        return " ".join(y)

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

    positive_count = np.count_nonzero(pred == 1)
    negative_count = np.count_nonzero(pred == 0)

    st.text("Total number of tweets:")
    st.write(len(pred))
    st.text("Percentage of positive tweets:")
    st.write(positive_count / len(pred) * 100)
    st.text("Percentage of positive tweets:")
    st.write(negative_count / len(pred) * 100)

    labels = ['Positive', 'Negative']
    counts = [positive_count, negative_count]

