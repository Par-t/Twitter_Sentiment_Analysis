# Twitter Sentiment Analysis

This project is a Python application that fetches real-time tweets based on user-entered hashtags and performs sentiment analysis on those tweets. It utilizes the Twitter API for tweet retrieval, a logistic regression model for sentiment analysis, and Streamlit for the frontend and hosting services. The project also includes a dataset sourced from Kaggle  training the classification model. Check out the live application[here](https://par-t-twitter-sentiment-analysis-main-streamlit-dbxpo2.streamlit.app/).

## Features

- Fetches real-time tweets based on user-entered hashtags
- Performs sentiment analysis on English tweets out of the fetched 100 tweets
- Utilizes a logistic regression model for sentiment analysis
- Provides a user-friendly frontend interface using Streamlit
- Hosted on a hosting service to make it accessible online

## Dataset

The sentiment analysis model has been trained using the "Sentiment140" dataset, which can be found on Kaggle at the following link: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140). The dataset contains 1.6 million tweets that have been labeled as positive or negative. For training the model, a sample of 40,000 tweets has been used.


## 'master' Branch
The master branch contains two Jupyter notebooks:

Classification_model.ipynb: This notebook focuses on preprocessing the training and testing data. It explores and evaluates different classification models using various metrics to identify the best model for the sentiment analysis task.

Real_time.ipynb: This notebook demonstrates the process of fetching live tweets using the Twitter API. It provides an example of how to retrieve real-time data for sentiment analysis.

The models and vectorizers obtained from these notebooks are converted into pickle files for later use.

## 'streamlit' Branch

The streamlit branch is dedicated to the front end and implementation of the sentiment analysis process using Streamlit. It utilizes the pickled models and vectorizers obtained from the master branch. The main code for the frontend, vectorization, and classification of tweets is implemented here. The Streamlit application allows users to enter hashtags and fetch real-time tweets to perform sentiment analysis.

# Local Deployment

To deploy and run the Twitter Sentiment Analysis project locally, follow the steps below:

1. Download all the files from the `streamlit` branch of the repository.

2. Install the required dependencies as started in the 'requirements.txt' file

3. If you want to retrain the classification model, you can do so by following the instructions in the `Classification_model.ipynb` notebook available in the `master` branch. After retraining the model, save it as a pickle file.

4. Replace the existing pickle files in the `streamlit` branch with your newly trained model pickle file(s). Make sure to update the code in the Streamlit application (`main.py`) to load your new model pickle file(s).

5. Create a new directory named .streamlit in the root of the project.

6. Inside the .streamlit directory, create a file named secrets.toml.

7. Open the secrets.toml file and add your Twitter API keys in the following format

CONSUMER\_KEY = "YOUR\_CONSUMER\_KEY"
CONSUMER\_SECRETt = "YOUR\_CONSUMER\_SECRET"
ACCESS\_TOKEN = "YOUR\_ACCESS\_TOKEN"
ACCESS\_TOKEN\_SECRET= "YOUR\_ACCESS\_TOKEN\_SECRET"
BEARER="YOUR\_BEARER\_TOKEN"

8. Run the Streamlit application by executing : streamlit run main.py

9. The Twitter Sentiment Analysis application should now be up and running locally.

Remember to make the necessary modifications and adjustments to the code as needed to suit your specific setup and requirements.

Please note that the "Sentiment140" dataset is not required for local deployment. It is used only for training and evaluating the sentiment analysis models.