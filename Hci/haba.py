import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, sosfilt
from pywt import wavedec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import tkinter as tk

# Load data
train_v = pd.read_csv('Train-V.csv', header=None)
train_h = pd.read_csv('Train-H.csv', header=None)
test_v = pd.read_csv('Test-V.csv', header=None)
test_h = pd.read_csv('Test-H.csv', header=None)
def preprocess_signals(data, lowcut=1.0, highcut=20.0, fs=176, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered = sosfilt(sos, data)
    normalized = (filtered - np.min(filtered, axis=0)) / (np.max(filtered, axis=0) - np.min(filtered, axis=0))
    return normalized - np.mean(normalized, axis=0)


import numpy as np
from scipy.signal import find_peaks
from pywt import wavedec


def extract_features(data, use_wavelets=False):
    features = []
    for signal in data.T:
        # Calculate raw signal features
        raw_features = [
            np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
            np.sum(np.square(signal)), len(find_peaks(signal)[0]), len(find_peaks(-signal)[0])
        ]
        if use_wavelets:
            # Perform wavelet decomposition
            coeffs = wavedec(signal, 'db4', level=2)
            # Assuming you are only interested in the first set of coefficients
            first_coeff = coeffs[0]
            # Append statistics of the first coefficient
            wavelet_features = [
                np.mean(first_coeff), np.std(first_coeff),
                np.min(first_coeff), np.max(first_coeff)
            ]
            # Combine raw features and wavelet features
            combined_features = np.concatenate([raw_features, wavelet_features])
        else:
            combined_features = raw_features

        features.append(combined_features)
    return np.array(features)




# Applying preprocessing
X_train_v = preprocess_signals(train_v.iloc[:-1, :])
X_train_h = preprocess_signals(train_h.iloc[:-1, :])
X_test_v = preprocess_signals(test_v)
X_test_h = preprocess_signals(test_h)

# Extracting features from V
X_train_features_raw_v = extract_features(X_train_v, use_wavelets=False)
X_train_features_wave_v = extract_features(X_train_v, use_wavelets=True)
X_test_features_raw_v = extract_features(X_test_v, use_wavelets=False)
X_test_features_wave_v = extract_features(X_test_v, use_wavelets=True)
# Extracting features from H
X_train_features_raw_H = extract_features(X_train_h, use_wavelets=False)
X_train_features_wave_H = extract_features(X_train_h, use_wavelets=True)
X_test_features_raw_H = extract_features(X_test_h, use_wavelets=False)
X_test_features_wave_H = extract_features(X_test_h, use_wavelets=True)
# Combining and scaling features
scaler = StandardScaler()
X_train_raw = scaler.fit_transform(np.concatenate((X_train_features_raw_v, X_train_features_raw_H), axis=0))
X_test_raw = scaler.transform(np.concatenate((X_test_features_raw_v, X_test_features_raw_H), axis=0))
X_train_wave = scaler.fit_transform(np.concatenate((X_train_features_wave_v, X_train_features_wave_H), axis=0))
X_test_wave = scaler.transform(np.concatenate((X_test_features_wave_v, X_test_features_wave_H), axis=0))

# Labels (assuming labels are provided in the last row of the training data)
y_train = np.concatenate((train_v.iloc[-1, :].values, train_h.iloc[-1, :].values))
print("y tarin Labels", y_train)
y_test = np.array([4] * 5 + [1] * 5 + [3] * 5 + [2] * 5 + [0] * 5+[4] * 5 + [1] * 5 + [3] * 5 + [2] * 5 + [0] * 5)
print("y test labels", y_test)
# Grid Search for best parameters
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}

grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=3)
grid_svm.fit(X_train_raw, y_train)
print("Best SVM Params:", grid_svm.best_params_)

grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=3)
grid_rf.fit(X_train_wave, y_train)
print("Best RF Params:", grid_rf.best_params_)

# Evaluate and compare
svm_best = grid_svm.best_estimator_
rf_best = grid_rf.best_estimator_
y_pred_svm_raw = svm_best.predict(X_test_raw)
y_pred_rf_wave = rf_best.predict(X_test_wave)

print("SVM Accuracy on raw features:", accuracy_score(y_test, y_pred_svm_raw))
print("RF Accuracy on wavelet features:", accuracy_score(y_test, y_pred_rf_wave))

# UI to display the detected direction
def show_direction(direction):
    window = tk.Tk()
    window.title("EOG Direction Detection")
    directions = ['Up', 'Down', 'Left', 'Right', 'Blink']  # Assuming 'Blink' or similar is a valid output
    for dir in directions:
        label = tk.Label(window, text=dir, font=("Arial", 24))
        label.pack(pady=10)
        if dir == direction:
            label.config(bg='green')
    window.mainloop()

# Assuming your model outputs integers that map to directions
predicted_direction = 'Right'  # Placeholder for actual model output
show_direction(predicted_direction)

# Save models
joblib.dump(svm_best, 'svm_best.joblib')
joblib.dump(rf_best, 'rf_best.joblib')
