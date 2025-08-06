import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

DATA_FOLDER = "data"
IMAGE_SIZE = (224, 224)
RANDOM_STATE = 42

mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    features = mobilenet.predict(x, verbose=0)
    return features.flatten()

def load_dataset(folder_path):
    X, y = [], []
    class_labels = os.listdir(folder_path)

    for label in class_labels:
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue

        print(f"[+] Processing: {label}")
        for filename in tqdm(os.listdir(label_path)):
            file_path = os.path.join(label_path, filename)
            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Error with {file_path}: {e}")

    return np.array(X), np.array(y)

print("Image loading and feature extraction...")
X, y = load_dataset(DATA_FOLDER)
print(f"Ready! Total examples: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

import joblib
joblib.dump(clf, "model.pkl")

y_pred = clf.predict(X_test)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Error matrix")
plt.show()

def predict_image(img_path, model):
    features = extract_features(img_path).reshape(1, -1)
    return model.predict(features)[0]

