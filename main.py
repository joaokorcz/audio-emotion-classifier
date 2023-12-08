import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    stft = np.abs(librosa.stft(y))

    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr,fmin=0.5*sr* 2**(-6)).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y),sr=sr*2).T,axis=0)
    features = np.hstack([mfccs,chroma,mel,contrast])

    return np.array(features)

def load_data(root_directory):
    features, labels = [], []

    emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']

    for emotion in emotions:
        qt = 0
        emotion_dir = os.path.join(root_directory, emotion)
        for filename in os.listdir(emotion_dir):
            qt += 1
            if qt == 100:
                break
            print(emotion, qt)
            if filename.endswith(".wav"):
                audio_path = os.path.join(emotion_dir, filename)
                feature = extract_features(audio_path)
                features.append(feature)
                labels.append(emotion)

    return np.array(features), np.array(labels)

root_directory = "./dataset"

# Carregando dados
features, labels = load_data(root_directory)

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Padronizando os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinando o modelo SVM
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Fazendo previsões
predictions = model.predict(X_test)

predictions_rf = model_rf.predict(X_test)

# Avaliando o desempenho
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"Acurácia: {accuracy}")
print("Relatório de Classificação:")
print(report)

accuracy_rf = accuracy_score(y_test, predictions_rf)
report_rf = classification_report(y_test, predictions_rf)

print(f"Acurácia: {accuracy_rf}")
print("Relatório de Classificação:")
print(report_rf)
