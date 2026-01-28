# YData EDA Analysis - EEG Sleep Emotion Decoder

## Overview
Comprehensive Exploratory Data Analysis (EDA) using YData Profiling for the EEG Sleep Emotion Decoder dataset. This analysis includes statistical profiling, visualizations, correlation analysis, and quality metrics.

## Dataset Characteristics

### Basic Statistics
- **Total Samples**: 280 EEG trials
- **Total Features**: 195 (192 numerical + 3 categorical)
- **Subjects**: 14 unique subjects (S_2, S_3, S_4, S_5, S_6, S_8, S_9, S_10, S_11, S_13, S_14, S_15, S_16, S_17)
- **EEG Channels**: 32 (standard EEG configuration)
- **Feature Types**: Statistical features extracted per channel:
  - Mean
  - Standard Deviation
  - Minimum Value
  - Maximum Value
  - Skewness
  - Kurtosis

### Emotion Label Distribution
- **Neutral**: 35.0% (98 samples)
- **Sleep**: 35.0% (98 samples)
- **Emotional**: 30.0% (84 samples)

## Data Quality Assessment

### Completeness
- **Missing Values**: 0 (100% complete)
- **Duplicate Rows**: 0 (100% unique)
- **Data Integrity**: Perfect

### Outlier Detection (IQR Method)
- **Total Outliers Detected**: 458
- **Affected Features**: 113 out of 193 numeric features
- **Top Outlier Features**:
  - Ch0_Mean: 30 outliers
  - Ch1_Mean: 24 outliers
  - Ch5_Skew: 12 outliers
  - Ch31_Min: 11 outliers
  - Ch27_Max: 10 outliers

## Statistical Features Summary

### Mean Values Across All Features
- **Mean Feature Value**: 0.1425
- **Std Feature Value**: 0.2561
- **Min Feature Value**: -7.95
- **Max Feature Value**: 7.23

### Distribution Characteristics
- **Average Skewness**: 0.2850 (slightly right-skewed distributions)
- **Average Kurtosis**: 0.7893 (lightly tailed distributions)
- **Top Skewed Features**: Ch25_Kurt, Ch28_Max, Ch26_Max
- **Top Kurtotic Features**: Ch25_Kurt, Ch1_Std, Ch0_Std

## Correlation Analysis

### Key Findings
- **Highly Correlated Pairs** (|r| > 0.9): 892 pairs
- **Moderately Correlated Pairs** (0.7 < |r| < 0.9): 1,599 pairs
- **Low Correlated Pairs** (|r| < 0.7): Majority of remaining pairs

### Top Correlated Feature Pairs
| Feature 1 | Feature 2 | Correlation |
|-----------|-----------|-------------|
| Ch0_Std | Ch1_Std | 0.9988 |
| Ch0_Std | Ch2_Std | 0.9986 |
| Ch1_Std | Ch2_Std | 0.9984 |
| Ch0_Std | Ch3_Std | 0.9980 |
| Ch0_Std | Ch4_Std | 0.9980 |

**Insight**: Standard deviation features across channels show very high correlation, suggesting redundancy and potential for dimensionality reduction (e.g., PCA).

## Visualizations Generated

### 1. Distribution Visualizations
**File**: `01_Feature_Distributions.png`
- 12 histogram plots showing feature distributions
- Covers first 12 channels with Mean, Std, Min, Max, Skew, Kurt metrics
- Shows approximately normal distributions with some skewness

**File**: `02_Feature_BoxPlots.png`
- 12 box plots for outlier visualization
- Identifies outliers per feature
- Shows spread and IQR for each feature

### 2. Correlation Analysis
**File**: `03_Correlation_Heatmap_Full.png`
- Full 193x193 correlation matrix heatmap
- Shows complex inter-feature relationships
- Predominantly blue diagonal pattern indicates strong within-channel correlations

**File**: `04_Correlation_Heatmap_Top20.png`
- Annotated 20x20 subset of top features
- Shows detailed correlation coefficients
- More interpretable visualization of key relationships

### 3. Data Quality
**File**: `05_Missing_Values_Analysis.png`
- Missing values per sample (all zero)
- Missing values per feature (all zero)
- Confirms 100% data completeness

### 4. Multivariate Analysis
**File**: `06_Scatter_Matrix.png`
- Pair plots of top 4 features
- Shows diagonal histograms
- Reveals linear relationships between features

**File**: `07_Subject_Emotion_Distribution.png`
- Bar chart: 20 samples per subject (balanced)
- Pie chart: 35%, 35%, 30% emotion distribution

### 5. Distribution Shape Analysis
**File**: `08_Skewness_Kurtosis.png`
- Top 10 features by skewness (Ch25_Kurt most skewed)
- Top 10 features by kurtosis (Ch25_Kurt highest)
- Indicates departure from normality in specific features

### 6. Interactive Visualizations
**Files**: `09_Interactive_Histogram.html`, `10_Interactive_Scatter.html`, `11_Interactive_BoxPlot.html`
- Plotly-based interactive charts
- Hover functionality for detailed inspection
- HTML format for web browser viewing

## Key Insights & Recommendations

### 1. Feature Redundancy
**Finding**: Standard deviation features show correlation > 0.99 between channels
**Recommendation**: 
- Apply dimensionality reduction (PCA) to reduce from 192 to ~50 principal components
- Consider channel-wise feature selection to reduce redundancy
- Aggregate features across channels

### 2. Data Quality
**Finding**: Perfect data completeness, no missing values, no duplicates
**Recommendation**: 
- No data cleaning required for missing values
- Data is ready for modeling as-is
- Consider handling 458 detected outliers based on domain knowledge

### 3. Emotion Classification Potential
**Finding**: Balanced emotion distribution (35%, 35%, 30%)
**Recommendation**:
- No class imbalance issue
- Good baseline for multi-class classification (0: Neutral, 1: Sleep, 2: Emotional)
- Can use standard stratified train-test splits

### 4. Feature Engineering
**Finding**: Extracted 6 statistical measures per channel (Mean, Std, Min, Max, Skew, Kurt)
**Recommendation**:
- Current features capture signal characteristics well
- Consider adding:
  - Frequency domain features (FFT, PSD)
  - Time-domain features (energy, entropy)
  - Cross-channel features (coherence, correlation)

### 5. Outlier Handling
**Finding**: 458 outliers detected, mainly in Min/Max values
**Recommendation**:
- Use robust scaling or outlier clipping for modeling
- Consider whether outliers represent valid physiological states
- Use algorithms robust to outliers (Random Forest, XGBoost)

## Modeling Recommendations

1. **Preprocessing Pipeline**:
   - Standardization (StandardScaler) - features have different ranges
   - Dimensionality reduction (PCA ~95% variance) to handle redundancy
   - Optional: Outlier clipping using IQR method

2. **Algorithms to Try**:
   - Random Forest (robust to outliers and scaling)
   - XGBoost/LightGBM (handles high-dimensional data well)
   - SVM with RBF kernel (after standardization)
   - Neural Networks (with proper scaling)

3. **Validation Strategy**:
   - Stratified K-Fold (preserve emotion distribution)
   - Cross-subject validation (test on held-out subject)

## Generated Files Summary

| File | Type | Purpose |
|------|------|---------|
| `01_Feature_Distributions.png` | PNG | Feature distribution overview |
| `02_Feature_BoxPlots.png` | PNG | Outlier detection |
| `03_Correlation_Heatmap_Full.png` | PNG | Full correlation matrix |
| `04_Correlation_Heatmap_Top20.png` | PNG | Detailed top features correlation |
| `05_Missing_Values_Analysis.png` | PNG | Data completeness check |
| `06_Scatter_Matrix.png` | PNG | Multivariate relationships |
| `07_Subject_Emotion_Distribution.png` | PNG | Data distribution by subject/emotion |
| `08_Skewness_Kurtosis.png` | PNG | Distribution shape analysis |
| `09_Interactive_Histogram.html` | HTML | Interactive feature distribution |
| `10_Interactive_Scatter.html` | HTML | Interactive feature scatter plot |
| `11_Interactive_BoxPlot.html` | HTML | Interactive emotion comparison |

## Conclusion

The EEG Sleep Emotion dataset is well-prepared with:
- ✅ Perfect data quality (no missing values, no duplicates)
- ✅ Balanced emotion distribution
- ✅ Sufficient samples (280 trials across 14 subjects)
- ✅ Rich feature representation (192 numerical features)
- ⚠️ High feature redundancy (correlation > 0.99)
- ⚠️ Some outliers detected (458 total)

The dataset is ready for machine learning with proper preprocessing to handle redundancy and scaling. The provided EDA visualizations enable data exploration and inform feature engineering decisions.

---

**Analysis Date**: January 27, 2026
**Tools Used**: YData Profiling, Pandas, Matplotlib, Seaborn, Plotly, SciPy
**Notebook**: `YData_EDA_Analysis.ipynb`
