# EEG Emotional Memory Pipeline - Advanced Integration Guide

## Overview

This guide demonstrates how to integrate **Transformer-based models** from Hugging Face with your EEG classification pipeline for enhanced performance.

## Architecture

```
Raw EEG Signal (16 channels × 200 timepoints)
    ↓
┌───────────────────────────────────────────────────────┐
│           Multiple Feature Extraction                   │
├───────────────────────────────────────────────────────┤
│                                                         │
│  1. Time Domain          2. Frequency Domain           │
│     ├─ Raw Power            ├─ Hilbert Transform      │
│     └─ Signal Envelope      └─ Instantaneous Power    │
│                                                         │
│  3. Transformer-Based    4. Riemannian Geometry       │
│     ├─ Zero-Shot            ├─ Covariance Matrix     │
│     ├─ Embeddings           └─ Eigenvalue Analysis   │
│     └─ Classification                                 │
│                                                         │
└───────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────┐
│           Enhanced Ensemble Classifier                 │
│  (Weighted combination: 25% each component)            │
└───────────────────────────────────────────────────────┘
    ↓
Time-Resolved Predictions (200 predictions per trial)
    ↓
Post-Processing (Window-based AUC optimization)
    ↓
Final Submission
```

## Transformer Integration

### 1. Zero-Shot Classification

Instead of training a classifier on predefined classes, zero-shot classifiers can evaluate your data against any set of labels without explicit training data.

```python
from transformers import pipeline

zero_shot_classifier = pipeline("zero-shot-classification", 
                                 model="facebook/bart-large-mnli")

# Classify EEG features
result = zero_shot_classifier(
    "high theta power with stable envelope",
    candidate_labels=["emotional_memory", "neutral_memory", "artifact"]
)
# Result: {
#   'labels': ['emotional_memory', 'neutral_memory', 'artifact'],
#   'scores': [0.45, 0.35, 0.20]
# }
```

**Advantages:**
- No training data needed for new classes
- Transfer learning from large language models
- Interpretable predictions based on semantic similarity

### 2. Feature Embeddings

Use transformer models to convert signal descriptions into rich feature embeddings:

```python
from transformers import AutoTokenizer, AutoModel

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Get embeddings for signal descriptions
description = "high theta power with variable envelope"
inputs = tokenizer(description, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # Shape: (1, seq_len, 768)
```

**Benefits:**
- Semantic understanding of signal characteristics
- 768-dimensional feature space
- Compatible with any downstream classifier

## Usage Example

### Basic Setup

```python
# Initialize feature extractor
feature_extractor = TransformerEEGFeatureExtractor()

# Initialize enhanced ensemble
ensemble = EnhancedEnsembleClassifier(weights={
    'time_domain': 0.25,
    'frequency_domain': 0.25,
    'transformer': 0.25,
    'riemannian': 0.25
})

# Get predictions
for eeg_signal in test_signals:
    prediction = ensemble.predict(eeg_signal)  # Returns: 0-1 probability
    print(f"Emotional probability: {prediction:.3f}")
```

### Custom Weights

```python
# Emphasize transformer features if they perform better
custom_ensemble = EnhancedEnsembleClassifier(weights={
    'time_domain': 0.15,
    'frequency_domain': 0.15,
    'transformer': 0.50,      # Increased weight
    'riemannian': 0.20
})
```

## Advanced Features

### 1. Signal Characteristic Detection

Automatically generate descriptions of EEG signal properties:

```python
extractor = TransformerEEGFeatureExtractor()
description = extractor.extract_signal_description(eeg_signal)
print(f"Signal characteristics: {description}")
# Output: "high power with stable envelope, max amplitude 45.23"
```

### 2. Multi-Label Classification

Classify signals across multiple dimensions simultaneously:

```python
# Classify along multiple axes
power_classification = zero_shot_classifier(
    description, 
    ["high power", "medium power", "low power"]
)

stability_classification = zero_shot_classifier(
    description,
    ["stable", "variable", "noisy"]
)

# Combine results for richer understanding
```

### 3. Domain Adaptation

Use transformer models to bridge domain gaps across subjects:

```python
# Describe each subject's signal characteristics
subject_descriptions = [
    extract_signal_description(subject_data) 
    for subject_data in all_subjects
]

# Compare semantic similarity across subjects
# This helps identify generalizable patterns
```

## Performance Optimization

### 1. Weight Optimization

Find optimal ensemble weights through cross-validation:

```python
from scipy.optimize import minimize

def cross_validate_weights(weights, X_val, y_val):
    ensemble = EnhancedEnsembleClassifier(weights=weights)
    predictions = [ensemble.predict(x) for x in X_val]
    # Compute AUC and return negative (for minimization)
    return -compute_auc(predictions, y_val)

result = minimize(
    cross_validate_weights,
    x0={'time_domain': 0.25, 'frequency_domain': 0.25, 
        'transformer': 0.25, 'riemannian': 0.25},
    args=(X_val, y_val)
)
optimal_weights = result.x
```

### 2. Feature Selection

Use transformer embeddings with feature selection:

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Extract embeddings for all samples
embeddings = feature_extractor.extract_transformer_embeddings(descriptions)

# Select top-k most discriminative features
selector = SelectKBest(f_classif, k=100)
best_features = selector.fit_transform(embeddings, y_train)
```

### 3. Model Stacking

Stack multiple classifiers on transformer features:

```python
from sklearn.ensemble import StackingClassifier

# Base learners
base_learners = [
    ('transformer_svm', SVC(probability=True)),
    ('transformer_rf', RandomForestClassifier()),
    ('transformer_lr', LogisticRegression())
]

# Meta learner
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression()
)

stacking_clf.fit(embeddings_train, y_train)
predictions = stacking_clf.predict_proba(embeddings_test)
```

## Installation & Requirements

```bash
# Basic requirements
pip install numpy scipy pandas scikit-learn

# Transformer support (optional but recommended)
pip install transformers torch

# Additional utilities
pip install scipy scikit-signal librosa
```

## Troubleshooting

### Issue: Transformers import fails
```bash
pip install --upgrade transformers torch
```

### Issue: CUDA out of memory
```python
# Use smaller model
model = "distilbert-base-uncased"  # Instead of bert-base-uncased

# Or use CPU only
import torch
torch.device('cpu')
```

### Issue: Slow inference
```python
# Cache model for reuse
from functools import lru_cache

@lru_cache(maxsize=128)
def get_classifier():
    return pipeline("zero-shot-classification")
```

## References

- **Transformers Library**: https://huggingface.co/transformers/
- **Zero-Shot Learning**: Brown et al., 2020 (GPT-3 paper)
- **Sentence Transformers**: https://www.sbert.net/
- **EEG-TCNet**: Musallam et al., 2021
- **Riemannian Geometry for EEG**: Barachant et al., 2013

## Key Takeaways

1. **Transformer Integration**: Leverage pre-trained models for feature extraction without domain-specific training data
2. **Zero-Shot Learning**: Classify signals into arbitrary categories without explicit training examples
3. **Ensemble Strength**: Combining multiple feature extraction methods improves robustness
4. **Domain Transfer**: Semantic embeddings help bridge gaps between subjects and conditions
5. **Flexibility**: Easily swap components or adjust weights for different datasets

---

**Last Updated**: January 24, 2026
**Status**: Production Ready
**Python Version**: 3.8+
