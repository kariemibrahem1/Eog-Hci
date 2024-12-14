import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, resample
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pywt import wavedec
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from imblearn.pipeline import make_pipeline
# Load the datasets
train_v = pd.read_csv('Train-V.csv', header=None)
train_h = pd.read_csv('Train-H.csv', header=None)
test_v = pd.read_csv('Test-V.csv', header=None)
test_h = pd.read_csv('Test-H.csv', header=None)

# Print first few rows of original data
print("Original Train-V Data (first 5 rows):")
print(train_v.head())

print("\nOriginal Train-H Data (first 5 rows):")
print(train_h.head())

print("\nOriginal Test-V Data (first 5 rows):")
print(test_v.head())

print("\nOriginal Test-H Data (first 5 rows):")
print(test_h.head())

# Separate labels
labels_train_v = train_v.iloc[-1, :]
labels_train_h = train_h.iloc[-1, :]
labels_test_v = test_v.iloc[-1, :]
labels_test_h = test_h.iloc[-1, :]

# Remove labels from the data
train_v = train_v.iloc[:-1, :]
train_h = train_h.iloc[:-1, :]
test_v = test_v.iloc[:-1, :]
test_h = test_h.iloc[:-1, :]

# Define Butterworth bandpass filter


# Define normalization function
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

# Existing normalization and DC component removal functions
def normalize(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def remove_dc(data):
    return data - np.mean(data, axis=0)

# Define parameters
lowcut = 1.0
highcut = 20.0
fs = 250  # Assume a sampling rate of 250 Hz
order = 2

# Apply the filter to the data
filtered_train_v = butter_bandpass_filter(train_v.values, lowcut, highcut, fs, order)
filtered_train_h = butter_bandpass_filter(train_h.values, lowcut, highcut, fs, order)
filtered_test_v = butter_bandpass_filter(test_v.values, lowcut, highcut, fs, order)
filtered_test_h = butter_bandpass_filter(test_h.values, lowcut, highcut, fs, order)

# Normalize the data
normalized_train_v = normalize(filtered_train_v)
normalized_train_h = normalize(filtered_train_h)
normalized_test_v = normalize(filtered_test_v)
normalized_test_h = normalize(filtered_test_h)

# Remove DC component
processed_train_v = remove_dc(normalized_train_v)
processed_train_h = remove_dc(normalized_train_h)
processed_test_v = remove_dc(normalized_test_v)
processed_test_h = remove_dc(normalized_test_h)

# Convert processed data back to DataFrame
processed_train_v_df = pd.DataFrame(processed_train_v)
processed_train_h_df = pd.DataFrame(processed_train_h)
processed_test_v_df = pd.DataFrame(processed_test_v)
processed_test_h_df = pd.DataFrame(processed_test_h)

# Add labels back to the DataFrame
processed_train_v_df.loc[len(processed_train_v_df)] = labels_train_v
processed_train_h_df.loc[len(processed_train_h_df)] = labels_train_h


# Save processed data to new CSV files
processed_train_v_df.to_csv('Processed-Train-V.csv', header=False, index=False)
processed_train_h_df.to_csv('Processed-Train-H.csv', header=False, index=False)
processed_test_v_df.to_csv('Processed-Test-V.csv', header=False, index=False)
processed_test_h_df.to_csv('Processed-Test-H.csv', header=False, index=False)

# Print first few rows of processed data
print("\nProcessed Train-V Data (first 5 rows):")
print(processed_train_v_df.head())

print("\nProcessed Train-H Data (first 5 rows):")
print(processed_train_h_df.head())

print("\nProcessed Test-V Data (first 5 rows):")
print(processed_test_v_df.head())

print("\nProcessed Test-H Data (first 5 rows):")
print(processed_test_h_df.head())



from scipy.signal import butter, filtfilt

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load processed data
processed_train_v_path = "Processed-Train-V.csv"
processed_train_h_path = "Processed-Train-H.csv"
processed_test_v_path = "Processed-Test-V.csv"
processed_test_h_path = "Processed-Test-H.csv"

train_v_df = pd.read_csv(processed_train_v_path, header=None)
train_h_df = pd.read_csv(processed_train_h_path, header=None)
test_v_df = pd.read_csv(processed_test_v_path, header=None)
test_h_df = pd.read_csv(processed_test_h_path, header=None)

# Extract features and labels
X_train_v = train_v_df.iloc[:-1, :].T  # All rows except the last one, transpose to have samples as rows
y_train_v = train_v_df.iloc[-1, :].values  # Last row as labels

X_train_h = train_h_df.iloc[:-1, :].T  # All rows except the last one, transpose to have samples as rows
y_train_h = train_h_df.iloc[-1, :].values  # Last row as labels

# Check if labels match
if not np.array_equal(y_train_v, y_train_h):
    raise ValueError("Training labels for vertical and horizontal data do not match")

# Assuming labels are the same, use one set of labels for training
y_train = y_train_v

X_test_v = test_v_df.T
X_test_h = test_h_df.T



import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from pywt import wavedec
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump, load


def extract_features(X):
    features = []
    # Ensure X is an ndarray and iterate properly
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)  # Reshape if it's a single sample
    for x in X:
        # Ensure x is a flat array for feature calculations
        x = np.ravel(x)
        # Statistical features
        mean = np.mean(x)
        variance = np.var(x)
        std_dev = np.std(x)
        energy = np.sum(x ** 2)
        # Morphological features
        peaks, _ = find_peaks(x)
        valleys, _ = find_peaks(-x)
        peak_pos = peaks[np.argmax(x[peaks])] if peaks.size > 0 else None
        valley_pos = valleys[np.argmax(-x[valleys])] if valleys.size > 0 else None
        area_under_curve = np.trapz(x)
        # Append features
        features.append([mean, variance, std_dev, energy, len(peaks), len(valleys), peak_pos, valley_pos, area_under_curve])
    return np.array(features)

# Extract features from training and test data
X_train_features_v = extract_features(X_train_v)
X_train_features_h = extract_features(X_train_h)
X_test_features_v = extract_features(X_test_v)
X_test_features_h = extract_features(X_test_h)

# Combine vertical and horizontal features after feature extraction
X_train_combined = np.concatenate((X_train_features_v, X_train_features_h), axis=1)
X_test_combined = np.concatenate((X_test_features_v, X_test_features_h), axis=1)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test_combined)

# Train and evaluate classifiers
def train_and_evaluate_classifier(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

# SVM classifier
svm_clf = SVC(kernel='linear', C=0.01)
y_pred_svm = train_and_evaluate_classifier(svm_clf, X_train_scaled, y_train, X_test_scaled)

# Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=40)
y_pred_rf = train_and_evaluate_classifier(rf_clf, X_train_scaled, y_train, X_test_scaled)

# Compare classifiers
def compare_classifiers(y_test, y_pred1, y_pred2):
    accuracy1 = accuracy_score(y_test, y_pred1)
    accuracy2 = accuracy_score(y_test, y_pred2)
    cm1 = confusion_matrix(y_test, y_pred1)
    cm2 = confusion_matrix(y_test, y_pred2)
    return accuracy1, accuracy2, cm1, cm2

# Output results
print("Predicted labels using SVM:")
print(y_pred_svm)
print("\nPredicted labels using Random Forest:")
print(y_pred_rf)

# Actual labels (based on description)
actual_labels = np.array([4] * 5 + [1] * 5 + [3] * 5 + [2] * 5 + [0] * 5)
print("Actual Labels", actual_labels)

# Calculate accuracy
accuracy_svm = accuracy_score(actual_labels, y_pred_svm)
accuracy_rf = accuracy_score(actual_labels, y_pred_rf)
print(f'SVM Accuracy: {accuracy_svm:.2f}')
print(f'Random Forest Accuracy: {accuracy_rf:.2f}')


import joblib  # Import joblib for saving models

# Define the function to save the model
def save_model(model, filename):
    joblib.dump(model, filename)

# After training and evaluating classifiers
y_pred_svm = train_and_evaluate_classifier(svm_clf, X_train_scaled, y_train, X_test_scaled)
y_pred_rf = train_and_evaluate_classifier(rf_clf, X_train_scaled, y_train, X_test_scaled)

# Save the models
save_model(svm_clf, 'svm_classifier.joblib')
save_model(rf_clf, 'random_forest_classifier.joblib')

# Comparison and results output
accuracy1, accuracy2, cm1, cm2 = compare_classifiers(actual_labels, y_pred_svm, y_pred_rf)
print("Confusion Matrix for SVM:\n", cm1)
print("Confusion Matrix for Random Forest:\n", cm2)

# If you wish to load the model later for predictions in your application or GUI
def load_model(filename):
    return joblib.load(filename)

# Example of loading the model
svm_model_loaded = load_model('svm_classifier.joblib')
rf_model_loaded = load_model('random_forest_classifier.joblib')

