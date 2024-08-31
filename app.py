import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt', force=True)
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Load the pre-trained models
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Title of the app
st.title('Email/SMS Spam Detection')

# 1- PREPROCESSING
def transforming_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization
    text = nltk.word_tokenize(text)
    
    # Removing special characters and stopwords
    y = [i for i in text if i.isalnum()]
    
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    
    return " ".join(y)

# Input for the SMS/Email text
input_sms = st.text_input('Enter your message')

# Transform the input text
if input_sms:
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


