import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import streamlit as st

## load the Lstm model
model = load_model("lstm_model.h5")

## load the toknizer
with open("tokenizer.pkl","rb") as f:
    tokenizer=pickle.load(f)


def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]

    token_list = pad_sequences([token_list],maxlen =max_sequence_len-1,padding="pre")
    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


## streamlit app
st.title("Next Word Prediction With LSTM")

input_text = st.text_input("Enter the sequence of word","to be or not to be")

if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next Word Prediction : {next_word}")



