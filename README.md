# Credit Default Risk Prediction

A comprehensive machine learning project for predicting credit default risk using Home Credit data. This project implements a complete end-to-end ML pipeline including exploratory data analysis, feature engineering, model development, and evaluation.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Models](#models)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)

## 🎯 Overview

This project aims to predict whether a loan applicant will default on their credit obligations. The solution employs advanced feature engineering techniques and ensemble machine learning algorithms to build a robust credit risk assessment model.

**Key Objectives:**
- Analyze credit default patterns through comprehensive EDA
- Engineer features from multiple data sources
- Build and compare various machine learning models
- Optimize model performance for imbalanced classification
- Evaluate model performance using appropriate metrics

## 📁 Project Structure

```
Credit_Default_Risk/
├── EDA.ipynb                                    # Exploratory Data Analysis
├── Preprocessing_FeatureEngineering_Clean.ipynb # Data preprocessing and feature engineering
├── Modeling.ipynb                               # Model training and hyperparameter tuning
├── Evaluation_Analysis.ipynb                    # Model evaluation and performance analysis
└── README.md                                    # Project documentation
```

## 📊 Dataset

The project uses the **Home Credit Default Risk** dataset, which includes:

- **Application data**: Client information at the time of application
- **Bureau data**: Credit history from other financial institutions
- **Previous applications**: Historical application data with Home Credit
- **Credit card balance**: Monthly credit card balance data
- **POS cash balance**: Point of sale and cash loan data
- **Installments payments**: Payment history for previous credits

The target variable indicates whether a client defaulted (1) or not (0) on their loan.

## 🔄 Workflow

### 1. Exploratory Data Analysis (`EDA.ipynb`)
- **Data Loading & Inspection**: Understanding data structure, types, and basic statistics
- **Missing Value Analysis**: Identifying and analyzing patterns in missing data
- **Target Distribution**: Analyzing class imbalance in the target variable
- **Feature Distributions**: Univariate analysis of numerical and categorical features
- **Correlation Analysis**: Identifying relationships between features and target
- **Visualization**: Creating comprehensive plots for data understanding
  - Distribution plots for numerical features
  - Count plots for categorical features
  - Correlation heatmaps
  - Box plots for outlier detection

### 2. Preprocessing & Feature Engineering (`Preprocessing_FeatureEngineering_Clean.ipynb`)
- **Data Loading**: Loading all relevant data tables
- **Data Cleaning**:
  - Handling missing values using imputation strategies
  - Removing outliers
  - Correcting data inconsistencies
- **Feature Engineering**:
  - Polynomial features creation
  - Aggregation features from related tables
  - Domain-specific feature creation
  - Interaction features
- **Data Transformation**:
  - Label encoding for categorical variables
  - Feature scaling and normalization
- **Feature Selection**:
  - Correlation-based feature selection
  - Removing low-variance features
- **Data Export**: Saving processed datasets for modeling

### 3. Model Development (`Modeling.ipynb`)
- **Data Preparation**:
  - Loading preprocessed data
  - Train-test split
  - Handling class imbalance
- **Baseline Models**:
  - Logistic Regression
  - Random Forest Classifier
- **Advanced Models**:
  - XGBoost (Extreme Gradient Boosting)
  - LightGBM (Light Gradient Boosting Machine)
- **Hyperparameter Tuning**:
  - Grid search for optimal parameters
  - Cross-validation for robust evaluation
- **Model Training**:
  - Training multiple models with optimized parameters
  - Feature importance analysis
- **Model Saving**: Exporting trained models for deployment

### 4. Evaluation & Analysis (`Evaluation_Analysis.ipynb`)
- **Performance Metrics**:
  - ROC-AUC Score (primary metric)
  - Precision, Recall, F1-Score
  - Confusion Matrix
- **Model Comparison**: Comparing all trained models
- **Feature Importance**: Analyzing which features contribute most to predictions using (MDI/Gini Impurity or Gain)
- **Error Analysis**: Understanding model mistakes and limitations

## 🤖 Models

The project implements and compares the following machine learning algorithms:

1. **Random Forest Classifier**
   - Ensemble of decision trees
   - Handles non-linear relationships well
   - Provides feature importance

2. **XGBoost (Extreme Gradient Boosting)**
   - Advanced gradient boosting algorithm
   - Excellent performance on structured data
   - Built-in regularization to prevent overfitting

3. **LightGBM (Light Gradient Boosting Machine)**
   - Fast and efficient gradient boosting
   - Handles large datasets efficiently
   - Lower memory usage

All models are optimized for the ROC-AUC metric, which is appropriate for imbalanced classification problems.

## 🛠️ Requirements

```python
# Core Libraries
numpy
pandas

# Visualization
matplotlib
seaborn
plotly

# Machine Learning
scikit-learn
xgboost
lightgbm

# Utilities
warnings
gc
```

## 💻 Usage

### Step 1: Exploratory Data Analysis
```bash
# Open and run EDA.ipynb to understand the data
jupyter notebook EDA.ipynb
```

### Step 2: Data Preprocessing and Feature Engineering
```bash
# Run preprocessing to prepare data for modeling
jupyter notebook Preprocessing_FeatureEngineering_Clean.ipynb
```

### Step 3: Model Training
```bash
# Train and optimize machine learning models
jupyter notebook Modeling.ipynb
```

### Step 4: Model Evaluation
```bash
# Evaluate model performance and analyze results
jupyter notebook Evaluation_Analysis.ipynb
```

**Note**: Update the file paths in the notebooks according to your local directory structure before running.

## 📈 Results

The models are evaluated using:
- **ROC-AUC Score**: Primary metric for ranking predictions
- **Precision & Recall**: Understanding trade-offs in classification
- **Confusion Matrix**: Analyzing true/false positives and negatives
- **Feature Importance**: Identifying key predictors of default risk

(Specific results will be populated after running all notebooks)

## 🎓 Academic Context

This project is developed as part of the **M.Tech Semester Coursework** in Data Science in Practise (DSIP) - Mini Project.

## 📝 Notes

- The project handles class imbalance through appropriate evaluation metrics (ROC-AUC)
- Feature engineering significantly improves model performance
- LightGBM and XGBoost typically outperform traditional models on this dataset
- Cross-validation ensures robust model evaluation
- All notebooks include detailed comments for reproducibility

---

**Repository**: [Credit_Default_Risk](https://github.com/tush1507/Credit_Default_Risk)  
**Branch**: main