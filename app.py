import streamlit as st
import pickle
import nltk
nltk.download('punkt', force=True)
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Initialize the PorterStemmer
ps = PorterStemmer()

# Load the pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Title of the app
st.title('Email/SMS Spam Detection')

# 1- PREPROCESSING
def transforming_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization
    text = nltk.word_tokenize(text)
    
    # Removing special characters and stopwords
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # Stemming
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)

# Input for the SMS/Email text
input_sms = st.text_input('Enter your message')

# Transform the input text
transformed_sms = transforming_text(input_sms)

if st.button('Predict'):
    # 2. VECTORIZE the input
    vector_input = tfidf.transform([transformed_sms])

    # 3. PREDICT
    result = model.predict(vector_input)

    # 4. DISPLAY the result
    if result == 1:
        st.header('SPAM')
    else:
        st.header('NOT SPAM')


