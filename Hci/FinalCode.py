import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import random
from scipy.signal import butter, sosfilt
from pywt import wavedec
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from imblearn.pipeline import make_pipeline

def load_data(filepath):
    """ Load EOG data from a file. """
    return pd.read_csv(filepath, header=None, sep="\t")

def preprocess_data(data):
    """ Apply a bandpass filter to preprocess the EOG data. """
    sos = butter(1, [0.05, 5], 'bandpass', fs=100, output='sos')
    filtered = sosfilt(sos, data.squeeze())
    return filtered

def extract_features(data):
    """ Extract statistical features and wavelet coefficients from the data. """
    features = [np.mean(data), np.std(data), np.min(data), np.max(data)]
    coeffs = wavedec(data, 'db4', level=2)
    coff=coeffs[0]
    features.extend([np.mean(coff) ] + [np.std(coff)])
    return np.array(features)

def load_and_process_data(directory_path, mode='train'):
    """ Load and process data from each subfolder and assign labels based on subfolder names. """
    all_features = []
    all_labels = []
    categories = ['Up', 'Blink', 'Right', 'Left', 'Down']
    for category in categories:
        for orientation in ['H', 'V']:
            folder_path = os.path.join(directory_path, f"{category}-{orientation}", f"{mode}-v")
            print(f"Processing folder: {folder_path}")
            if not os.path.exists(folder_path):
                print(f"Folder not found: {folder_path}")
                continue
            for filename in os.listdir(folder_path):
                filepath = os.path.join(folder_path, filename)
                data = load_data(filepath)
                processed_data = preprocess_data(data)
                features = extract_features(processed_data)
                all_features.append(features)
                all_labels.append(category)
    if not all_features:
        raise ValueError("No data found. Please check the directory paths and structure.")
    return np.array(all_features), np.array(all_labels)

def train_classifiers(X_train, y_train, X_test, y_test):
    """Train classifiers with handling for class imbalance and proper data splitting."""
    models = {
        "SVM": SVC(kernel='rbf', probability=True),
        "Random Forest": RandomForestClassifier()
    }
    param_grids = {
        "SVM": {'C': [1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 0.01]},
        "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20, None]}
    }

    best_models = {}
    for name, model in models.items():
        # Incorporating SMOTE for balancing within the cross-validation loop
        pipeline = make_pipeline(SMOTE(random_state=42),
                                 GridSearchCV(model, param_grids[name], cv=StratifiedKFold(n_splits=5),
                                              scoring='accuracy'))
        pipeline.fit(X_train, y_train)
        best_estimator = pipeline.named_steps['gridsearchcv'].best_estimator_
        best_models[name] = best_estimator

        # Evaluation
        y_pred = best_estimator.predict(X_test)
        print(f"{name} Best GridSearchCV Score: {pipeline.named_steps['gridsearchcv'].best_score_}")
        print(f"{name} Test Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")

    return best_models

def save_model(model, filename):
    """ Save the trained model to a file. """
    dump(model, filename)

def setup_gui(model_path):
    """ Setup the GUI for real-time EOG direction prediction. """
    model = load(model_path)
    root = tk.Tk()
    root.title("EOG Direction Detection")

    # Setup the GUI layout
    arrows = {
        "Left": tk.Label(root, text="←", font=("Arial", 44)),
        "Right": tk.Label(root, text="→", font=("Arial", 44)),
        "Up": tk.Label(root, text="↑", font=("Arial", 44)),
        "Down": tk.Label(root, text="↓", font=("Arial", 44)),
        "Blink": tk.Label(root, text="👁️", font=("Arial", 44))  # Eye icon for blink
    }

    # Grid positioning
    arrows["Up"].grid(row=0, column=1)
    arrows["Left"].grid(row=1, column=0)
    arrows["Blink"].grid(row=1, column=1)  # Central position for Blink
    arrows["Right"].grid(row=1, column=2)
    arrows["Down"].grid(row=2, column=1)

    def select_file_and_predict():
        """ Allow the user to select a file and predict the direction. """
        filepath = filedialog.askopenfilename()
        if filepath:
            data = load_data(filepath)
            processed_data = preprocess_data(data)
            features = extract_features(processed_data).reshape(1, -1)
            prediction = model.predict(features)[0]
            for arrow in arrows.values():
                arrow.config(fg="black")  # Reset all to default
            if prediction in arrows:
                arrows[prediction].config(fg="red")  # Highlight the predicted direction
            else:
                # Randomly choose a direction if 'Other' or undefined category
                chosen_direction = random.choice(list(arrows.keys()))
                arrows[chosen_direction].config(fg="red")

    tk.Button(root, text="Select File and Predict", command=select_file_and_predict).grid(row=3, column=0, columnspan=3)
    root.mainloop()

if __name__ == "__main__":
    directory_path = '3-class'

    # Load and process training data
    X_train, y_train = load_and_process_data(directory_path, mode='train')

    # Load and process testing data
    X_test, y_test = load_and_process_data(directory_path, mode='test')

    # Train classifiers
    models = train_classifiers(X_train, y_train, X_test, y_test)

    # Save the best model
    save_model(models['Random Forest'], 'random_forest_model.joblib')

    # Setup GUI
    setup_gui('random_forest_model.joblib')
