import os
import pandas as pd


def gather_data(class_dir, folder_name, signal_type, include_class_label=True):
    data = []
    class_labels = []
    classes = ['Blink', 'Down', 'Left', 'Right', 'Up']

    for class_name in classes:
        folder_path = os.path.join(class_dir, class_name, f"{class_name}-{signal_type}", folder_name)
        if os.path.exists(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                signal_data = pd.read_csv(file_path, header=None).squeeze("columns")
                data.append(signal_data)
                if include_class_label:
                    class_labels.append(class_name)

    # Combine all signals into a DataFrame, with each signal as a column
    combined_data = pd.concat(data, axis=1)

    # Add class labels as a new row
    if include_class_label:
        class_row = pd.DataFrame([class_labels], columns=combined_data.columns)
        combined_data = pd.concat([combined_data, class_row], ignore_index=True)

    return combined_data


# Base directory containing the extracted dataset
class_dir = '3-class'

# Create Train-V.csv
train_v_data = gather_data(class_dir, 'train-v', 'V', include_class_label=True)
train_v_data.to_csv(os.path.join(class_dir, 'Train-V.csv'), index=False)

# Create Train-H.csv
train_h_data = gather_data(class_dir, 'train-v', 'H', include_class_label=True)
train_h_data.to_csv(os.path.join(class_dir, 'Train-H.csv'), index=False)

# Create Test-V.csv
test_v_data = gather_data(class_dir, 'test-v', 'V', include_class_label=False)
test_v_data.to_csv(os.path.join(class_dir, 'Test-V.csv'), index=False)

# Create Test-H.csv
test_h_data = gather_data(class_dir, 'test-v', 'H', include_class_label=False)
test_h_data.to_csv(os.path.join(class_dir, 'Test-H.csv'), index=False)

# Verify the files have been created
os.listdir(class_dir)
