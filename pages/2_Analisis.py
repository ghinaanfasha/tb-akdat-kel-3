import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("Analisis Data")

# Pastikan data yang telah diproses tersedia di session state
if 'preprocessed_data' in st.session_state and st.session_state.preprocessed_data is not None:
    data = st.session_state.preprocessed_data

    # Pisahkan fitur (X) dan label (y)
    X = data.drop(columns=['Heart Disease'])  # Pastikan Heart Disease tidak ada
    y = data['Heart Disease']  # Label target

    # Cek jika ada NaN di y (label), dan tangani
    if y.isnull().any():
        st.warning("Data label mengandung NaN, melakukan pembersihan data label.")
        data = data.dropna(subset=['Heart Disease'])  # Hapus NaN pada label
        y = data['Heart Disease']  # Update y setelah pembersihan

    # Bagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi dan latih model Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Simpan model di session state
    st.session_state.model = model

    # Formulir untuk input data baru
    st.subheader("Formulir Input Data Baru")
    # Input dari pengguna
    age = st.slider("Umur:", 0, 100, 50)
    sex = st.radio("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    chest_pain = st.selectbox("Tipe Nyeri Dada", [1, 2, 3, 4])
    bp = st.slider("Tekanan Darah:", 50, 200, 120)
    cholesterol = st.slider("Kolesterol:", 100, 400, 200)
    fbs_over_120 = st.radio("Gula Darah Puasa > 120", ["Tidak", "Ya"])
    ekg_results = st.selectbox("Hasil EKG", [0, 1, 2])
    max_hr = st.slider("Detak Jantung Maksimal:", 60, 200, 150)
    exercise_angina = st.radio("Angina Saat Olahraga", ["Tidak", "Ya"])
    st_depression = st.slider("Depresi ST:", 0.0, 5.0, 1.0)
    slope_of_st = st.selectbox("Kemiringan ST", [1, 2, 3])
    num_vessels = st.selectbox("Jumlah Pembuluh Darah Fluro", [0, 1, 2, 3])
    thallium = st.selectbox("Hasil Tes Thallium", [3, 6, 7])

    # Validasi input sebelum ditambahkan
    input_data = {
        "age": age,
        "sex": sex,
        "chest_pain": chest_pain,
        "bp": bp,
        "cholesterol": cholesterol,
        "fbs_over_120": fbs_over_120,
        "ekg_results": ekg_results,
        "max_hr": max_hr,
        "exercise_angina": exercise_angina,
        "st_depression": st_depression,
        "slope_of_st": slope_of_st,
        "num_vessels": num_vessels,
        "thallium": thallium
    }

    # Memastikan semua input diisi
    if all(v is not None for v in input_data.values()):
        # Mapping input ke nilai numerik
        sex = 1 if sex == "Laki-laki" else 0
        fbs_over_120 = 1 if fbs_over_120 == "Ya" else 0
        exercise_angina = 1 if exercise_angina == "Ya" else 0

        # DataFrame dari input pengguna tanpa kolom 'Heart Disease'
        user_data = pd.DataFrame([[age, sex, chest_pain, bp, cholesterol, fbs_over_120, 
                                   ekg_results, max_hr, exercise_angina, st_depression, 
                                   slope_of_st, num_vessels, thallium]], 
                                 columns=X.columns)  # Gunakan kolom dari X

        # Prediksi untuk data baru
        prediction = model.predict(user_data)
        predicted_label = prediction[0]

        # Menambahkan hasil prediksi sebagai kolom 'Heart Disease'
        user_data['Heart Disease'] = predicted_label

        # Menampilkan data input pengguna sebelum ditambahkan
        st.write("Data yang akan ditambahkan:")
        st.write(user_data)

        # Tambahkan data baru ke dataset yang ada
        if st.button("Tambahkan Data Baru"):
            # Menambahkan data baru ke session state
            st.session_state.preprocessed_data = pd.concat([data, user_data], ignore_index=True)

            # Menampilkan data setelah penambahan baris baru dengan scrollable table
            st.write("Data Setelah Input Baru:")
            st.dataframe(st.session_state.preprocessed_data, use_container_width=True)  # Menampilkan data terbaru dengan scroll

        # Prediksi untuk data baru
        if st.button("Prediksi Penyakit Jantung"):
            result = "Presence" if predicted_label == 1 else "Absence"
            st.write(f"Prediksi: Penyakit Jantung {result}")

    else:
        st.warning("Harap lengkapi semua kolom input sebelum menambahkan data baru.")
else:
    st.write("Belum ada data yang diproses.")
