# EEG Sleep-Emotion Decoder: Hybrid Ensemble Pipeline

A comprehensive deep learning and Riemannian Geometry hybrid pipeline for EEG-based Emotional Memory Classification with time-resolved predictions.

## 📌 Project Overview

This repository contains a state-of-the-art (SOTA) solution for the **EEG Emotional Memory Classification Challenge**. The pipeline achieves high performance by combining:

1. **Deep Learning (Temporal Focus)**: Modified EEG-TCNet with dense prediction output for capturing temporal dynamics
2. **Riemannian Geometry (Spatial Focus)**: Tangent Space Mapping + SVM for capturing covariance structure and spatial patterns

## 🚀 Key Features

- **Hybrid Ensemble**: Combines temporal (TCN) and spatial (Riemannian) approaches
- **Advanced Preprocessing**: Bandpass filtering (0.5-40 Hz) + Euclidean Alignment (EA)
- **Dense Predictions**: Outputs probability for each timepoint (200 predictions per trial)
- **Subject-Invariant**: Euclidean Alignment ensures zero-shot generalization across subjects
- **Attention Mechanisms**: Multi-scale temporal convolutions with attention gates
- **Post-processing**: Gaussian smoothing for continuous high-probability windows
- **Ensemble Optimization**: Weighted averaging (60% TCN + 40% Riemannian)

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/EEG-Sleep-Emotion-Decoder.git](https://github.com/YourUsername/EEG-Sleep-Emotion-Decoder.git)
    cd EEG-Sleep-Emotion-Decoder
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Usage

1.  **Prepare Data:**
    Download the competition `.mat` files and place them in:
    * `./data/training/`
    * `./data/testing/`

2.  **Run the Pipeline:**
    ```bash
    python src/main.py
    ```

3.  **Output:**
    The script will generate `submission.csv` containing probability predictions for each `{subject}_{trial}_{timepoint}`.

## 📊 Results
* **Target Metric:** Window-Based AUC.
* **Validation Strategy:** Leave-One-Group-Out (LOGO) cross-validation to ensure zero-shot generalization capabilities.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
