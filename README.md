# 🕸️ TrackPulse  
### ML-based Song Hit Predictor + Similar Track Recommender

TrackPulse is a machine learning project that predicts whether a song is likely to become a hit based on its audio features and recommends similar tracks using a k-Nearest Neighbors model.  
The app is built with Streamlit for a simple, interactive user interface.

---

## 🚀 Features
- Predicts **Hit vs Not Hit**
- Displays **hit probability**
- Recommends **similar songs** using kNN
- Shows **similarity score**
- Uses custom engineered features
- Easy-to-use Streamlit UI

---

## 🧠 How It Works
1. User enters audio features for a song  
2. Custom combined features are generated:
   - `energy * danceability`
   - `valence * loudness`
   - `tempo * energy`
3. Values are scaled using saved `StandardScaler` objects  
4. Classification model predicts hit/not hit  
5. kNN model finds nearest songs from the dataset  
6. UI displays predictions and recommendations

---

## 📂 Project Structure
TrackPulse/
│
├── ui.py # Streamlit UI
├── hit_model.pkl # Classification model
├── rec_model.pkl # kNN recommendation model
├── hit_scaler.pkl # Scaler for hit prediction
├── rec_scaler.pkl # Scaler for recommendation
├── dataset.csv # Audio features dataset
├── requirements.txt
└── README.md

---

## 🛠 Tech Stack
- Python
- Streamlit
- Scikit-Learn
- Pandas / NumPy
- Joblib

---

## ▶️ Run Locally

### 1. Install dependencies
pip install -r requirements.txt

### 2. Start the Streamlit app
streamlit run app.py

---

## 📦 Requirements
streamlit
pandas
scikit-learn
joblib
numpy

---

## 📄 License
MIT License
