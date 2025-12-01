import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")

hit_model = joblib.load('hit_model.pkl')
hit_scaler = joblib.load('hit_scaler.pkl')
rec_model = joblib.load('rec_model.pkl')
rec_scaler = joblib.load('rec_scaler.pkl')

df = pd.read_csv('dataset.csv')

st.title('🕸️TrackPulse')

st.subheader('Want to check if your song would be a hit or not? Enter your song features below!')
st.subheader('Happy predicting!')

with st.expander("🎵 Basic Song Info", expanded=True):
    song_name = st.text_input("Song Name")
    duration_ms = st.number_input("Duration (ms)", 10000, 600000, 180000)
    explicit = st.selectbox("Explicit (0 = No, 1 = Yes)", [0, 1])

    key_options = {"C":0,"C# / Db":1,"D":2,"D# / Eb":3,"E":4,"F":5,"F# / Gb":6,
                   "G":7,"G# / Ab":8,"A":9,"A# / Bb":10,"B":11}
    key_name = st.selectbox("Key", list(key_options), index=5)
    key = key_options[key_name]

    mode = st.selectbox("Mode (0=Minor, 1=Major)", [0,1], index=1)

with st.expander("🎚️ Audio Features", expanded=True):
    danceability = st.number_input("Danceability", 0.0, 1.0, 0.75, step=0.01)
    energy = st.number_input("Energy", 0.0, 1.0, 0.82, step=0.01)
    loudness = st.number_input("Loudness (dB)", -60.0, 0.0, -5.3, step=0.1)
    tempo = st.number_input("Tempo (BPM)", 50, 250, 120)
    speechiness = st.number_input("Speechiness", 0.0, 1.0, 0.05, step=0.01)
    acousticness = st.number_input("Acousticness", 0.0, 1.0, 0.10, step=0.01)
    instrumentalness = st.number_input("Instrumentalness", 0.0, 1.0, 0.0, step=0.01)
    liveness = st.number_input("Liveness", 0.0, 1.0, 0.12, step=0.01)
    valence = st.number_input("Valence", 0.0, 1.0, 0.65, step=0.01)

user_input = {
    "duration_ms": duration_ms,
    "explicit": explicit,
    "key": key,
    "mode": mode,
    "speechiness": speechiness,
    "acousticness": acousticness,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "tempo": tempo,
    "energy": energy,
    "danceability": danceability,
    "loudness": loudness,
    "valence": valence
}

user_input['energy_dance'] = user_input['energy'] * user_input['danceability']
user_input['valence_loudness'] = user_input['valence'] * user_input['loudness']
user_input['tempo_energy'] = user_input['tempo'] * user_input['energy']

# print(user_input)

hit_features = [
    "duration_ms", "explicit", "key", "mode", "speechiness", 
    "acousticness", "instrumentalness", "liveness", "tempo", 
    "energy_dance", "valence_loudness", "tempo_energy"
]

rec_features = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo"
]

user_hit = {}
for i in hit_features:
    user_hit[i]=user_input[i]

# print(user_hit)

hit_df = pd.DataFrame([user_hit])

# print(hit_df)


user_rec = {}
for i in rec_features:
    user_rec[i]=user_input[i]

# print(user_rec)

rec_df = pd.DataFrame([user_rec])

# print(rec_df)

hit_scaled = hit_scaler.transform(hit_df)
rec_scaled = rec_scaler.transform(rec_df)

hit_pred = hit_model.predict(hit_scaled)[0] # predict() → gives array predict()[0] → gives actual value
# print(hit_pred)
hit_prob = hit_model.predict_proba(hit_scaled)[0, 1]  # probability of hit

distances,indices = rec_model.kneighbors(rec_scaled,n_neighbors=4)

if st.button('Predict Hit or Not'):

    if song_name=='':

        st.warning("Please enter a song name!")

    else:

        hit_result = f"{song_name} might be a HIT! 🥳" if hit_pred == 1 else f"{song_name} probably won't be a hit 😅"
        st.success(hit_result)

        # st.info(f"Hit probability: {hit_prob*100:.2f}%")
        st.progress(hit_prob)
        st.caption(f"Hit Probability: {hit_prob*100:.2f}%")

        st.subheader("🎧 Similar Songs Recommendations")

        for id in indices[0][1:]:
            row = df.iloc[id]
            with st.expander(f"{row['track_name']} — {row['artists']} 🎵"):
                st.write(f"Tempo: {row['tempo']} BPM")
                st.write(f"Danceability: {row['danceability']}")
                st.write(f"Energy: {row['energy']}")
                st.write(f"Liveness: {row['liveness']}")
                st.write(f"Valence: {row['valence']}")

                similarity_score = (1 - distances[0][list(indices[0]).index(id)]) * 100
                st.progress(similarity_score / 100)
                st.caption(f"Similarity Score: {similarity_score:.2f}%")