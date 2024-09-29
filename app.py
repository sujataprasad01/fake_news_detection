import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import nltk

nltk.download('stopwords')

news_df = pd.read_csv('train.csv.zip')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']

ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_df['content'] = news_df['content'].apply(stemming)

X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)

st.sidebar.title("Recent Articles")
recent_articles = []  

def add_article(article):
    if article not in recent_articles:
        recent_articles.append(article)
    if len(recent_articles) > 5: 
        recent_articles.pop(0)

st.title('Fake News Detector')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    add_article(input_text) 
    if pred == 1:
        st.write('The News is Fake')
    else:
        st.write('The News Is Real')

if recent_articles:
    st.sidebar.subheader("Recently Analyzed Articles")
    for article in recent_articles:
        st.sidebar.write(article)

