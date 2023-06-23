# Twitter Sentiment Analysis

This project is a Python application that fetches real-time tweets based on user-entered hashtags and performs sentiment analysis on those tweets. It utilizes the Twitter API for tweet retrieval, a logistic regression model for sentiment analysis, and Streamlit for the frontend and hosting services. The project also includes a dataset sourced from Kaggle  training the classification model.

## Features

- Fetches real-time tweets based on user-entered hashtags
- Performs sentiment analysis on English tweets out of the fetched 100 tweets
- Utilizes a logistic regression model for sentiment analysis
- Provides a user-friendly frontend interface using Streamlit
- Hosted on a hosting service to make it accessible online

## Requirements

The requirements have been listed in 'requirements.txt' in the streamlit branch and the jupyter notebooks in the master branch.

## Dataset

The sentiment analysis model has been trained using the "Sentiment140" dataset, which can be found on Kaggle at the following link: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140). The dataset contains 1.6 million tweets that have been labeled as positive or negative. For training the model, a sample of 40,000 tweets has been used.

