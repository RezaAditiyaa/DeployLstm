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

# Streamlit UI
st.title("🧠 Analisis Sentimen Indonesia Emas 2045")

# Input teks
text = st.text_area("Masukkan teks untuk dianalisis:")

if st.button("Prediksi Sentimen"):
    if text:
        try:
            processed_text = text_to_sequence(text, tokenizer)
            probability_sentiment = model.predict(processed_text)[0][0]
            
            # Aturan baru: Positif jika lebih dari 0.8
            predicted_sentimen_label = 'Positif' if probability_sentiment > 0.8 else 'Negatif'

            # Tampilkan hasil
            st.write(f"### **Prediksi Sentimen: {predicted_sentimen_label}**")
            st.write(f"📊 **Probabilitas Sentimen:** {round(probability_sentiment, 4)}")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")
