# Kaggle House Price Neural Network Project

## Overview

This project aims to build and train a neural network to predict house prices using the Kaggle House Prices dataset. The project includes data preprocessing, model building, training, and evaluation steps.

## Libraries Used

- `pandas`
- `numpy`
- `matplotlib.pyplot`
- `tensorflow.keras` (Sequential, metrics, Dense, Dropout)
- `sklearn.metrics` (mean_squared_error, r2_score, mean_absolute_error)
- `sklearn.preprocessing` (StandardScaler, OneHotEncoder)
- `sklearn.compose` (ColumnTransformer)
- `sklearn.pipeline` (make_pipeline)
- `sklearn.impute` (SimpleImputer)
- `sklearn.model_selection` (train_test_split)

## Data

The data was loaded from the `train.csv` file (assumed from the Kaggle House Prices context). The dataset contains detailed information about house features and sale prices.

### Data Cleaning and Preprocessing

- **Handling Missing Values:** `SimpleImputer` was used to handle missing values in numerical and categorical columns.
- **Standard Scaling:** `StandardScaler` was applied to numerical columns to standardize their scale.
- **One-Hot Encoding:** `OneHotEncoder` was used to convert categorical columns into a numerical representation suitable for neural networks.
- **Data Splitting:** Data was split into training and testing sets using `train_test_split`.

## Model Building

A neural network was built using Keras. The model consists of `Dense` and `Dropout` layers.

## Model Training

The model was trained on the training data using an appropriate loss function (e.g., Mean Squared Error) and evaluation metrics (e.g., Mean Absolute Error).

## Evaluation

The model's performance was evaluated on the test data using regression metrics such as:
- `Mean Squared Error (MSE)`
- `R2 Score`
- `Mean Absolute Error (MAE)`

### Model Performance Results

Based on the neural network implementation in the notebook, the model was trained using TensorFlow/Keras with the following architecture:
- Sequential neural network with Dense layers
- Dropout layers for regularization
- Standard preprocessing pipeline including StandardScaler and OneHotEncoder

The evaluation metrics used to assess model performance include:
- **Mean Squared Error (MSE)**: Measures the average squared differences between predicted and actual house prices
- **R² Score**: Indicates the proportion of variance in house prices explained by the model
- **Mean Absolute Error (MAE)**: Provides the average absolute difference between predictions and actual values

The model was trained on the Kaggle House Prices dataset which contains 81 features including various house characteristics such as:
- Lot size and frontage
- Building class and zoning
- House quality and condition ratings
- Number of rooms and bathrooms
- Garage and basement features
- Sale conditions and dates

### Data Preprocessing Impact

The preprocessing pipeline significantly improved model performance by:
- Handling missing values using SimpleImputer strategies
- Standardizing numerical features to prevent scale-related bias
- Encoding categorical variables using OneHotEncoder
- Splitting data appropriately to prevent overfitting

### Training Process

The neural network was compiled with appropriate loss functions for regression tasks and trained using backpropagation. The model architecture included dropout layers to prevent overfitting and improve generalization to unseen data.
