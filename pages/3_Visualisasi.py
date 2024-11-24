import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Visualisasi Data Kesehatan Jantung")

if 'preprocessed_data' in st.session_state and st.session_state.preprocessed_data is not None:
    data = st.session_state.preprocessed_data

    st.subheader("Pengaruh Umur terhadap Risiko Penyakit Jantung")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Age', hue='Heart Disease', multiple='stack', kde=True, bins=20)
    plt.xlabel('Usia')
    plt.ylabel('Jumlah Pasien')
    plt.title('Pengaruh Umur terhadap Risiko Penyakit Jantung')
    st.pyplot(plt)

    st.subheader("Perbandingan Tekanan Darah dengan Tingkat Kolesterol terhadap Risiko Penyakit Jantung")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='BP', y='Cholesterol', hue='Heart Disease')
    plt.xlabel('Tekanan Darah')
    plt.ylabel('Tingkat Kolesterol')
    plt.title('Perbandingan Tekanan Darah dengan Tingkat Kolesterol terhadap Risiko Penyakit Jantung')
    st.pyplot(plt)

    st.subheader("Korelasi Faktor Usia, Tekanan Darah, Kolesterol, dan Denyut Jantung Maksimal terhadap Risiko Penyakit Jantung")
    correlation_matrix = data[['Age', 'BP', 'Cholesterol', 'Max HR', 'Heart Disease']].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis')
    plt.title('Korelasi Faktor Usia, Tekanan Darah, Kolesterol, dan Denyut Jantung Maksimal terhadap Risiko Penyakit Jantung')
    st.pyplot(plt)

else:
    st.write("Belum ada data yang diproses. Silakan kembali ke halaman input.")
