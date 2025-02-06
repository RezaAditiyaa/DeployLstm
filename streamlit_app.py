import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer dan model
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load model LSTM
model = load_model('lstm_model.h5')

# Fungsi untuk mengubah teks ke dalam urutan angka (sequences)
def text_to_sequence(text, tokenizer, max_len=100):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequence

# Fungsi untuk menentukan label sentimen
def sentimen_label_with_probability(prediction):
    probability_positive = prediction
    probability_negative = 1 - prediction
    
    sentiment_label = 'Positif' if probability_positive > 0.6 else 'Negatif'
    return sentiment_label, probability_positive, probability_negative

# Streamlit UI
st.title("🧠 Analisis Sentimen dengan LSTM")

# Input teks
text = st.text_area("Masukkan teks untuk dianalisis:")

if st.button("Prediksi Sentimen"):
    if text:
        try:
            processed_text = text_to_sequence(text, tokenizer)
            predicted_sentimen = model.predict(processed_text)[0][0]
            predicted_sentimen_label, prob_pos, prob_neg = sentimen_label_with_probability(predicted_sentimen)

            st.write(f"### **Prediksi Sentimen: {predicted_sentimen_label}**")
            st.write(f"📊 **Probabilitas Positif:** {round(prob_pos, 4)}")
            st.write(f"📉 **Probabilitas Negatif:** {round(prob_neg, 4)}")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")
