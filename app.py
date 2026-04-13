import streamlit as st
import numpy as np
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

## Load the trained model and tokenizer
model = load_model('next_word_lstm.h5')
with open('tokenizer.pickle','rb') as f:
    tokenizer = pickle.load(f)



# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence length
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    # Cast token_list to float32 or int32 for Keras 3 / TensorFlow compatibility
    token_list = np.array(token_list, dtype=np.int32)
    
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    # Use tokenizer index to map integer index back to text word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

## Streamlit app
st.title("Next Word Prediction with LSTM RNN")
input_text = st.text_input("Enter a sequence of words:", "to be or not to be")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # max_sequence_len is input length + 1 for the label
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    if next_word:
        st.write(f"Predicted next word: '{next_word}'")
    else:
        st.write("Could not predict the next word.")