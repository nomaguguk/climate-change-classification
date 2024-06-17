#for interactive app
import streamlit as st
#serialising objects 
import pickle 
#for experimental tracking
import comet_ml
from comet_ml import Experiment 
#preprocessing text data
import nltk
nltk.download('stopwords')
from unidecode import unidecode
from nltk.corpus import stopwords
import spacy
#vectorisation
from sklearn.feature_extraction.text import TfidfVectorizer
#file management
import os
import re
#api
os.environ['API_KEY'] = 'gRDjhvM5j3nQFdkpLREGdZxJ3'

import warnings
warnings.filterwarnings(action = 'ignore')

#set experiment object 
experiment = Experiment(api_key=os.environ.get('API_KEY'),
                        project_name='climate-change-sentiment-analysis',
                        workspace='nomaguguk')

experiment.set_name('streamlit-run-demo')#

# Removing URLs, mentions, retweets and hashtags
# Use regular expressions to strip out unwanted parts of the text.
def tweet_filtering(message):
    '''
    Utility function to preprocess the tweets by removing special characters,
    accents, and digits

    Args:
        message (str): raw tweet to be preprocessed

    Returns:
        str: preprocessed tweet
    '''

    # Remove URLs
    message = re.sub(r"https?:\/\/.*\/\w*", ' ', message)

    # Remove mentions
    message = re.sub("@[\w]*", ' ', message)

    # Remove retweets
    message = re.sub(r"^\s*rt\s+", ' ', message)

    # Remove hashtags
    message = re.sub(r"#\w*", ' ', message)

    # Remove special characters
    message = re.sub(r"\W", ' ', message)

    # Remove digits
    message = re.sub(r"\d+", ' ', message)

    # Remove large whitespaces
    message = re.sub(r"\s\s+", ' ', message)

    # Remove leading and trailing whitespaces
    message = message.strip()

    # Remove accents
    message = unidecode(message)

    return message

#define stop words
stop_words = set(stopwords.words('english'))

# Remove stop words
def remove_stopwords(message):
    '''
    Remove stop words i.e. words which do not add meaning to text
    
    Args:
        message (str): raw text 
        
    Returns:
        str: filtered text without stopwords
    '''
    return ' '.join([word for word in message.split() if word not in stop_words])

#reduces words to their base or dictionary form 
nlp = spacy.load("en_core_web_sm")

# Lemmatize text
def lemmatize_text(message):
    '''
    This function takes raw text and reduces each word to its base/dictionary form
    
    Args:
        message (str): raw text to be processed
        
    Returns:
        str: lemmatized text
    '''
    doc = nlp(message)
    return ' '.join([token.lemma_ for token in doc])


#load vectorizer 
with open(r'\climate-change-sentiment-analysis-predict\objects\tfidf_vectorizer.pkl', 'rb') as f: 
    tfidf_vectorizer = pickle.load(f)


#load model 
with open(r'\climate-change-sentiment-analysis-predict\models\rf_best_model.pkl', 'rb') as m: 
    model = pickle.load(m)


#streamlit app 
def main():
    st.title('Climate Change Sentiment Analysis')
    tweet = st.text_input('Enter a tweet:')

    #sentiment dictionary
    sentiment_class_dict = {-1: 'Anti', 0: 'Neutral', 1: 'Pro', 2: 'Factual'}

    if st.button('Analyze Sentiment'):
        if tweet:
            #normalize tweet
            tweet = tweet.lower()
            #filter noise
            tweet = tweet_filtering(tweet)
            #remove stopwords
            tweet = remove_stopwords(tweet)
            #lemmatize
            tweet = lemmatize_text(tweet)
            #vectorize
            tweet_tfidf = tfidf_vectorizer.transform([tweet]).toarray()
            #prediction
            prediction = model.predict(tweet_tfidf)
            prediction_class = prediction.item()
            #define sentiment
            sentiment = sentiment_class_dict[prediction_class]
            #display output
            st.write('Sentiment:', sentiment)
            
            # Log the prediction to Comet.ml
            experiment.log_text("input_tweet", tweet)
            experiment.log_metric("predicted_sentiment", sentiment)
        else:
            st.write('Please enter a tweet text')
            
    experiment.end()

if __name__ == "__main__":
    main()