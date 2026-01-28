# Quick Reference: EEG Transformer Integration

## 🚀 Quick Start

```python
# 1. Load and preprocess data
X_train, y_train = load_and_preprocess_eeg()

# 2. Initialize transformer feature extractor
from transformers import pipeline
feature_extractor = TransformerEEGFeatureExtractor()

# 3. Create enhanced ensemble
ensemble = EnhancedEnsembleClassifier(weights={
    'time_domain': 0.25,
    'frequency_domain': 0.25,
    'transformer': 0.25,
    'riemannian': 0.25
})

# 4. Make predictions
predictions = [ensemble.predict(signal) for signal in X_test]

# 5. Generate submission
submission = generate_submission(predictions)
```

## 📊 Feature Extraction Methods

| Method | Description | Advantages | Computation |
|--------|-------------|------------|-------------|
| **Time Domain** | Raw signal power per channel | Fast, interpretable | O(n) |
| **Frequency Domain** | Hilbert transform instantaneous power | Preserves temporal info | O(n log n) |
| **Transformer** | Zero-shot classification + embeddings | Semantic understanding | O(n²) |
| **Riemannian** | Covariance eigenvalue analysis | Geometrically principled | O(n³) |

## 🎯 Zero-Shot Classification

```python
# Define candidate labels (no training data needed!)
labels = ["emotional_memory", "neutral_memory", "sleep_artifact"]

# Classify based on signal description
result = zero_shot_classifier(
    "high theta power with stable envelope",
    labels
)

# Extract probabilities
emotional_prob = result['scores'][0]
```

## 🔧 Configuration

### Default Weights (Balanced)
```python
weights = {
    'time_domain': 0.25,      # Raw power features
    'frequency_domain': 0.25,  # Hilbert transform
    'transformer': 0.25,       # Zero-shot classification
    'riemannian': 0.25         # Covariance analysis
}
```

### Transformer-Optimized Weights
```python
weights = {
    'time_domain': 0.15,
    'frequency_domain': 0.15,
    'transformer': 0.50,       # Emphasize transformer
    'riemannian': 0.20
}
```

### Frequency-Focused Weights
```python
weights = {
    'time_domain': 0.15,
    'frequency_domain': 0.50,  # Emphasize frequency domain
    'transformer': 0.15,
    'riemannian': 0.20
}
```

## 📈 Performance Metrics

```python
# Cross-validation
from sklearn.model_selection import cross_validate

cv_scores = cross_validate(
    ensemble,
    X_train, y_train,
    cv=5,
    scoring=['accuracy', 'roc_auc', 'f1']
)

print(f"Mean AUC: {cv_scores['test_roc_auc'].mean():.3f} ± {cv_scores['test_roc_auc'].std():.3f}")
```

## 🎨 Visualization

```python
import matplotlib.pyplot as plt

# Plot ensemble weights
methods = list(ensemble.weights.keys())
weights = list(ensemble.weights.values())

plt.figure(figsize=(10, 4))
plt.bar(methods, weights)
plt.ylabel('Weight')
plt.title('Ensemble Component Weights')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 💾 Saving & Loading

```python
import pickle

# Save trained ensemble
with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)

# Load for inference
with open('ensemble_model.pkl', 'rb') as f:
    ensemble = pickle.load(f)
```

## ⚡ Performance Tips

1. **GPU Acceleration**: Use CUDA for transformer models
   ```bash
   pip install torch torchvision torchaudio
   ```

2. **Batch Processing**: Process multiple signals at once
   ```python
   batch_predictions = [ensemble.predict(sig) for sig in batch]
   ```

3. **Caching**: Cache transformer models
   ```python
   @lru_cache(maxsize=1)
   def get_model():
       return AutoModel.from_pretrained("distilbert-base-uncased")
   ```

4. **Parallel Processing**: Use multiprocessing for data loading
   ```python
   from multiprocessing import Pool
   with Pool(4) as p:
       predictions = p.map(ensemble.predict, signals)
   ```

## 🐛 Debugging

```python
# Enable verbose output
import logging
logging.basicConfig(level=logging.DEBUG)

# Step through ensemble prediction
signal = X_test[0]
time_feat = ensemble.extract_time_domain_features(signal)
freq_feat = ensemble.extract_frequency_domain_features(signal)
trans_feat = ensemble.extract_transformer_features(signal)
rier_feat = ensemble.extract_riemannian_features(signal)

print(f"Time domain: {time_feat}")
print(f"Freq domain: {freq_feat}")
print(f"Transformer: {trans_feat}")
print(f"Riemannian: {rier_feat}")
```

## 📚 Key Files

- `notebooks/EEG_Emotional_Memory_Pipeline.ipynb` - Main notebook
- `src/preprocessing.py` - EEG preprocessing functions
- `src/models.py` - Model implementations
- `TRANSFORMER_INTEGRATION_GUIDE.md` - Detailed integration guide
- `requirements.txt` - Python dependencies

## 🔗 Resources

- **Transformers**: https://huggingface.co/transformers/
- **CLIP Model** (for image/signal classification): https://github.com/openai/CLIP
- **Zero-Shot Learning**: https://arxiv.org/abs/1803.06175
- **EEG-TCNet**: https://arxiv.org/abs/2006.00927

## ✅ Checklist Before Submission

- [ ] Data loaded and preprocessed ✓
- [ ] Feature extraction tested ✓
- [ ] Ensemble weights configured ✓
- [ ] Cross-validation passed ✓
- [ ] Window-based AUC computed ✓
- [ ] Submission CSV generated ✓
- [ ] Format validated ✓
- [ ] Ready for upload! 🎉

---

**Version**: 1.0.0 | **Date**: January 24, 2026
