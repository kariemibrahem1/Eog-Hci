#from Wavelet
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from pywt import wavedec
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
train_v = pd.read_csv('Train-V.csv', header=None)
train_h = pd.read_csv('Train-H.csv', header=None)
test_v = pd.read_csv('Test-V.csv', header=None)
test_h = pd.read_csv('Test-H.csv', header=None)

# Print original data
print("Original Train-V Data:\n", train_v.head())
print("Original Train-H Data:\n", train_h.head())
print("Original Test-V Data:\n", test_v.head())
print("Original Test-H Data:\n", test_h.head())

def preprocess_signals(data, lowcut=1.0, highcut=20.0, fs=176, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data, axis=0)
    normalized = (filtered - np.min(filtered, axis=0)) / (np.max(filtered, axis=0) - np.min(filtered, axis=0))
    return normalized - np.mean(normalized, axis=0)

# Apply preprocessing
X_train_v = preprocess_signals(train_v.iloc[:-1, :])
X_train_h = preprocess_signals(train_h.iloc[:-1, :])
X_test_v = preprocess_signals(test_v)
X_test_h = preprocess_signals(test_h)

# Print preprocessed data
print("Preprocessed Train-V Data:\n", X_train_v[:5])
print("Preprocessed Train-H Data:\n", X_train_h[:5])
print("Preprocessed Test-V Data:\n", X_test_v[:5])
print("Preprocessed Test-H Data:\n", X_test_h[:5])

def extract_features(data):
    features = []
    for signal in data.T:
        coeffs = wavedec(signal, 'db4', level=2)
        first_coeff = coeffs[0]
        features.append([
            np.mean(first_coeff), np.std(first_coeff), np.min(first_coeff), np.max(first_coeff),
            np.sum(first_coeff**2), len(find_peaks(first_coeff)[0]), len(find_peaks(-first_coeff)[0])
        ])
    return np.array(features)

# Feature extraction
X_train_features_v = extract_features(X_train_v)
X_train_features_h = extract_features(X_train_h)
X_test_features_v = extract_features(X_test_v)
X_test_features_h = extract_features(X_test_h)

# Print extracted features
print("Extracted Features Train-V:\n", X_train_features_v[:5])
print("Extracted Features Train-H:\n", X_train_features_h[:5])
print("Extracted Features Test-V:\n", X_test_features_v[:5])
print("Extracted Features Test-H:\n", X_test_features_h[:5])

# Combine and scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(np.concatenate((X_train_features_v, X_train_features_h), axis=0))
X_test_scaled = scaler.transform(np.concatenate((X_test_features_v, X_test_features_h), axis=0))

# Print scaled features
print("Scaled Train Features:\n", X_train_scaled[:5])
print("Scaled Test Features:\n", X_test_scaled[:5])
# Labels
y_train = np.concatenate((train_v.iloc[-1, :].values, train_h.iloc[-1, :].values))
print("y tarin Labels", y_train)
y_test = np.array([4] * 5 + [1] * 5 + [3] * 5 + [2] * 5 + [0] * 5+[4] * 5 + [1] * 5 + [3] * 5 + [2] * 5 + [0] * 5)
print("y test labels", y_test)
# Classifier training
svm_clf = SVC(kernel='rbf', C=10)
svm_clf.fit(X_train_scaled, y_train)
y_pred_svm = svm_clf.predict(X_test_scaled)

rf_clf = RandomForestClassifier(n_estimators=450, random_state=42)
rf_clf.fit(X_train_scaled, y_train)
y_pred_rf = rf_clf.predict(X_test_scaled)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Print classifier predictions
print("Predicted by SVM:\n", y_pred_svm)
print("Predicted by RandomForest:\n", y_pred_rf)
print("Predicted by KNN:\n", y_pred_knn)

# Evaluate and print results
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# Save the models
joblib.dump(svm_clf, 'svm_classifier_w.joblib')
joblib.dump(rf_clf, 'random_forest_classifier_w.joblib')
joblib.dump(knn, 'KNN_classifier_w.joblib')

import tkinter as tk
import joblib


def flash_arrow(direction):
    # Reset all labels to default state
    for key, label in labels.items():
        label.config(bg='white')

    # Flash the corresponding arrow in red
    if direction in labels:
        labels[direction].config(bg='red')


def create_ui():
    window = tk.Tk()
    window.title("EOG Direction Detection")

    # Create arrow labels
    global labels
    # Setup the GUI layout
    labels = {
        "Left": tk.Label(window, text="←", font=("Arial", 44)),
        "Right": tk.Label(window, text="→", font=("Arial", 44)),
        "Up": tk.Label(window, text="↑", font=("Arial", 44)),
        "Down": tk.Label(window, text="↓", font=("Arial", 44)),
        "Blink": tk.Label(window, text="*", font=("Arial", 44))  # Eye icon for blink
    }

    # Grid positioning
    labels["Up"].grid(row=0, column=1)
    labels["Left"].grid(row=1, column=0)
    labels["Blink"].grid(row=1, column=1)  # Central position for Blink
    labels["Right"].grid(row=1, column=2)
    labels["Down"].grid(row=2, column=1)


    return window


# Load your model and predict
def predict_and_display():
    # Load model
    model = joblib.load('random_forest_classifier_w.joblib')
    prediction = model.predict([X_test_scaled[0]])  # Example prediction on n test sample
    direction = directions[prediction[0]]  # Map your prediction to a direction
    flash_arrow(direction)


if __name__ == "__main__":
    directions = {0: 'Up', 1: 'Down', 2: 'Right', 3: 'Left', 4: 'Blink'}

    # Create and run the UI
    window = create_ui()
    # This should ideally be triggered by real-time data or in a loop with test data
    predict_and_display()

    window.mainloop()








