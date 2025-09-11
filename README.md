# Credit Card Fraud Detection: Machine Learning Analysis

A comprehensive end-to-end machine learning project for detecting credit card fraud using advanced algorithms and imbalanced data techniques.

## Project Overview

This project analyzes credit card transactions to build an effective fraud detection system using machine learning. The dataset contains 284,807 transactions from European cardholders with a highly imbalanced class distribution (0.172% fraud rate).

### Key Results
- **Best Model**: XGBoost (Balanced) achieving **86.61% AUC-PR**
- **Business Impact**: **$9,811 net benefit** per 56,962 transactions
- **Fraud Detection Rate**: **83.7%** (catches 82 out of 98 fraud cases)
- **False Alarm Rate**: **0.04%** (minimal customer disruption)

## Dataset Information

- **Size**: 284,807 transactions
- **Features**: 30 (Time, Amount, V1-V28 PCA-transformed)
- **Target**: Binary classification (0: Normal, 1: Fraud)
- **Class Distribution**: 284,315 normal vs 492 fraudulent (577:1 ratio)
- **Data Quality**: No missing values, clean dataset

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis
1. Place the `creditcard.csv` dataset in the project directory: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Open and run `credit_card_fraud_detection.ipynb` in Jupyter Notebook
3. The notebook will automatically handle all preprocessing, modeling, and evaluation

## Model Performance Comparison

| Model | AUC-PR | Precision | Recall | F1-Score |
|-------|--------|-----------|--------|----------|
| **XGBoost (Balanced)** | **0.8661** | **0.7810** | **0.8367** | **0.8079** |
| Random Forest (Balanced) | 0.8187 | 0.8182 | 0.8265 | 0.8223 |
| Neural Network + SMOTE | 0.8220 | 0.7034 | 0.8469 | 0.7685 |
| Baseline Logistic Regression | 0.7429 | 0.8289 | 0.6429 | 0.7241 |

## Key Technical Insights

### Class Imbalance Solutions Effectiveness

1. **Class Weighting (Winner)**
   - XGBoost `scale_pos_weight=577.9`
   - Maintains data integrity while adjusting model bias
   - **Result**: Optimal precision-recall balance

2. **SMOTE + Tree Models**
   - Creates synthetic fraud examples for balanced training
   - Works well with ensemble methods
   - **Result**: Higher recall, acceptable precision drop

3. **SMOTE + Linear Models**
   - Same synthetic data, wrong algorithm choice
   - **Result**: Precision collapse (3-6%) - avoid in production

### Feature Importance Discovery

- **V14 (PCA component)**: 56.3% importance - the "fraud smoking gun"
- **Amount**: Only 7th most important (1.86%) - fraud is about patterns, not amounts
- **Robustness**: Random Forest distributes importance more evenly

## Business Impact Analysis

### Financial Performance (Test Set: 56,962 transactions)
```
Fraud Prevented:    $10,021  (82 cases caught × $122 avg fraud)
Fraud Missed:       $1,955   (16 cases lost)
Investigation Cost: $210     (105 investigations × $2 cost)
────────────────────────────
Net Benefit:        $9,811   (Strong ROI)
```

### Operational Metrics
- **Customer Impact**: 23 false alarms out of 56,864 normal transactions
- **Investigation Workload**: 105 total cases (manageable)
- **Fraud Detection**: 84% catch rate (industry-competitive)

## Project Structure

```
Credit Card Fraud Detection/
├── README.md                              # This file
├── credit_card_fraud_detection.ipynb      # Main analysis notebook
├── creditcard.csv                         # Dataset (not included)
└── requirements.txt                       # Dependencies
```

## Technical Implementation

### Data Preprocessing
- **Scaling**: RobustScaler (resistant to outliers)
- **Train/Test Split**: 80/20 stratified split
- **Class Balancing**: Multiple strategies tested (SMOTE, undersampling, class weights)

### Model Pipeline
1. **Baseline**: Logistic Regression on original imbalanced data
2. **Balanced Linear Models**: LR with SMOTE, undersampling, SMOTEENN
3. **Tree-Based Models**: Random Forest and XGBoost with class balancing
4. **Neural Networks**: MLPClassifier with various balancing strategies

### Evaluation Framework
- **Primary Metric**: AUC-PR (appropriate for imbalanced data)
- **Secondary Metrics**: F1-Score, Precision, Recall
- **Business Metrics**: Financial impact, investigation workload
- **Visualization**: ROC curves, Precision-Recall curves, confusion matrices

## Key Visualizations

The notebook includes comprehensive visualizations:
- Class distribution analysis
- Feature correlation heatmaps
- Model performance comparisons
- ROC and Precision-Recall curves
- Feature importance rankings
- Business impact charts

## Methodology Highlights

### Why This Approach Works
- **Algorithm Selection**: Tree-based models excel at non-linear pattern recognition
- **Imbalance Handling**: Class weighting preserves data integrity better than oversampling
- **Evaluation Strategy**: AUC-PR focuses on minority class performance
- **Business Focus**: Metrics aligned with operational and financial outcomes

### Validation Strategy
- **Cross-Validation**: Stratified K-fold for reliable performance estimates
- **Temporal Validation**: Train/test split preserves time-based patterns
- **Multiple Metrics**: Comprehensive evaluation beyond single accuracy score

## Future Enhancements

- **Feature Engineering**: Create domain-specific fraud indicators
- **Ensemble Methods**: Combine multiple model predictions
- **Deep Learning**: Experiment with neural network architectures
- **Anomaly Detection**: Unsupervised methods for novel fraud patterns
- **Real-time Pipeline**: MLOps implementation with continuous monitoring

---

*This analysis demonstrates the critical importance of proper evaluation techniques and business-focused machine learning for fraud detection systems.*
