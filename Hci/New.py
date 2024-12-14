import os
import shutil
from sklearn.model_selection import train_test_split


def split_files(folder_path):
    files = os.listdir(folder_path)
    train_files, test_files = train_test_split(files, test_size=0.25, random_state=42)

    # Create train and test directories
    train_dir = os.path.join(folder_path, 'train-v')
    test_dir = os.path.join(folder_path, 'test-v')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move files to the respective directories
    for file in train_files:
        shutil.move(os.path.join(folder_path, file), os.path.join(train_dir, file))
    for file in test_files:
        shutil.move(os.path.join(folder_path, file), os.path.join(test_dir, file))


# Base directory containing the extracted dataset
class_dir = '3-class'

# Apply the function to each subfolder
subfolders = ['Blink', 'Down', 'Left', 'Right', 'Up']
for subfolder in subfolders:
    for suffix in ['-H', '-V']:
        folder_path = os.path.join(class_dir, subfolder, subfolder + suffix)
        split_files(folder_path)
