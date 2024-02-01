# Laptop Price Prediction System

## Introduction
This repository contains a machine learning-based system for predicting laptop prices. The system utilizes a dataset that simulates laptop prices, capturing various features commonly associated with laptops.

## Dataset
The dataset includes key attributes such as:
- Brand
- Processor
- Speed
- RAM size
- Storage
- Capacity
- Screen Size
- Weight

### Feature Engineering
Two additional features were created:
1. **Screen Size Category:** Categorized screen sizes as Small, Medium, or Large.
2. **Weight Category:** Classified weights as Light, Medium, or Heavy.

## Machine Learning Pipeline
A pipeline was implemented to evaluate different regression models. The following models were considered:
- Linear Regression
- Lasso Regression
- Ridge Regression

### Model Selection
Lasso Regression was chosen based on MSE, MAE, and R-squared scores.

### Model Evaluation
The Lasso Regression model exhibited high performance on both the training and test sets.

### Cross Validation
Cross-validation with 5 folds ensured the model's robustness.

### Feature Importance
Analysis revealed 'Storage Capacity' as the most influential feature.

## Model Persistence
The trained Lasso Regression model was saved for future use.

## Usage
1. Clone the repository.
2. Install the required dependencies.
3. Run the provided Jupyter notebook or use the saved Lasso Regression model for predictions.

## Files
- `Laptop_price.csv`: The simulated laptop prices dataset.
- `laptop.ipynb`: Jupyter notebook containing the entire pipeline.
- `lasso_model.joblib`: Saved Lasso Regression model.

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Joblib

