import streamlit as st
import pickle

import re
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stemmer = SnowballStemmer('english')


def transform_text(text):
    # Lowercase
    text = text.lower()

    # Keep only words and numbers
    tokens = re.findall(r'\b\w+\b', text)

    # Remove stopwords
    filtered = [word for word in tokens if word not in ENGLISH_STOP_WORDS]

    # Apply stemming
    stemmed = [stemmer.stem(word) for word in filtered]

    return " ".join(stemmed)


tf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Detection")

input_sen = st.text_input("Enter the messages")

if st.button('Predict'):

    transformed_sms = transform_text(input_sen)

    vector_input = tf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("spam")
    else:
        st.header("Not Spam")

