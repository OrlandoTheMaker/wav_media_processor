import librosa
import os

# Directory where you saved the downloaded audio files
data_dir = r'Enter directory path here'


def extract_features(file_path):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)

    return mfcc


# List to store the features and labels
features = []
labels = []

# Iterate over the audio files in the directory
for file_name in os.listdir(data_dir):
    if file_name.endswith(".wav"):
        file_path = os.path.join(data_dir, file_name)
        emotion_label = file_name.split("-")[2]  # Extract the emotion label from the file name

        # Extract MFCC features and store them along with the label
        with open("features.txt", 'wb')as wb:
            mfcc_features = extract_features(file_path)
            wb.write(mfcc_features)
            features.append(mfcc_features)
            labels.append(emotion_label)


# Now you have the features and labels ready for further processing and model training
# You can convert them to suitable data structures like NumPy arrays or pandas DataFrame
