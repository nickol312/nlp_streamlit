import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import numpy as np
import pandas as pd

import pickle

import streamlit as st

with open('Model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

model = load_model('Model/my_model.h5')

def welcome():
    return "Welcome All"

def predict(sentence ,tokenizer, max_length, padding_type, trunc_type):
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    print(model.predict(padded))
    return(model.predict(padded))
    
def main():
    st.title("NLP Sentiment Recognition")
    sente = st.text_area('Text to analyze',placeholder="Type Here")
    sentence = [sente]
    result=""
    
    if st.button("Predict"):
        st.subheader(sentence)
        prediction = predict(sentence ,tokenizer, max_length, padding_type, trunc_type)
        
        if(prediction>0.5):
            result = "Positive"
        else:
            result = "Negative"
        
    st.success('The output is {}'.format(result))
    
if __name__=='__main__':
    main()
