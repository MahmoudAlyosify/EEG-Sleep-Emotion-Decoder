# 🧠 EEG Sleep Emotion Decoder

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

## 📌 Project Overview
This repository contains a state-of-the-art (SOTA) solution for the **EEG Emotional Memory Classification Challenge**. The goal is to detect whether a neutral or emotional memory is being reactivated during NREM sleep based on 16-channel EEG signals.

The pipeline achieves high performance on the **Window-Based AUC metric** by combining Temporal Convolutional Networks (Deep Learning) with Riemannian Tangent Space Mapping (Geometry-based ML), utilizing **Euclidean Alignment** for robust zero-shot generalization across subjects.

## 🚀 Key Features
* **Hybrid Ensemble:** Combines the strengths of Deep Learning (Dense-TCN) and Riemannian Geometry (Tangent Space SVM/LR).
* **Subject-Invariant Alignment:** Implements **Euclidean Alignment (EA)** to align covariance matrices of all subjects to a common reference, solving the domain shift problem.
* **Dense Prediction:** Modified TCN architecture that outputs predictions for every timepoint (200Hz) without global pooling.
* **Metric Optimization:** Advanced post-processing using Gaussian Smoothing to maximize the specific window-based AUC scoring metric.

## 🛠️ Pipeline Architecture

1.  **Preprocessing:**
    * Bandpass Filter (0.5Hz - 40Hz).
    * Euclidean Alignment ($\tilde{X}_i = R^{-1/2} X_i$).
2.  **Model A (Temporal):**
    * Dense-TCN with Dilated Convolutions.
    * Input: `(16, 200)` -> Output: `(200, 1)`.
3.  **Model B (Spatial):**
    * Sliding Window Covariance Estimation.
    * Tangent Space Mapping + Logistic Regression.
4.  **Ensemble & Post-processing:**
    * Weighted Average.
    * Gaussian Smoothing ($\sigma=3.0$).

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
