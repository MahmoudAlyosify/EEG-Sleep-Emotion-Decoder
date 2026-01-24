# Data Directory

This directory contains the EEG dataset files (.mat files) for the Sleep-Emotion Decoder project.

## Dataset Structure

The dataset is organized into two categories:
- **training/** - Contains training data split by emotion and state
  - sleep_emo/ - Sleep with emotion label data
  - sleep_neu/ - Sleep with neutral label data
  
- **testing/** - Contains test data for model evaluation

## Downloading the Data

The `.mat` files are not included in the repository due to size constraints. 

### Steps to set up:
1. Download the EEG dataset from the source (check project documentation)
2. Extract the files to maintain the following structure:
   ```
   data/
   ├── training/
   │   ├── sleep_emo/
   │   │   ├── S_2_cleaned.mat
   │   │   ├── S_3_cleaned.mat
   │   │   └── ...
   │   └── sleep_neu/
   │       ├── S_2_cleaned.mat
   │       ├── S_3_cleaned.mat
   │       └── ...
   └── testing/
       ├── test_subject_1.mat
       ├── test_subject_7.mat
       └── ...
   ```

3. All Python scripts reference this directory structure automatically

## Data Format

Each `.mat` file contains EEG signal data preprocessed and formatted for model training and evaluation.
