# PCA Analysis

This script applies Principal Component Analysis (PCA) for dimensionality reduction on the Breast Cancer Wisconsin dataset, followed by Logistic Regression for binary classification.

## Features

- Imports the Breast Cancer Wisconsin dataset
- Performs PCA on the full dataset
- Plots the cumulative explained variance ratio
- Reduces data to 2 principal components
- Implements Logistic Regression on the reduced dataset
- Calculates and reports classification accuracy

## Installation

Install the required packages using pip:
```
pip install scikit-learn matplotlib numpy
```


## How to Run

Execute the script from the command line:

```
python pca.py
```

## Output

The script produces:
1. Printed component variance ratios
2. A plot of cumulative explained variance
3. The shape of the 2D-reduced dataset
4. Classification accuracy of the Logistic Regression model

## Customization

- Adjust the `n_components` parameter in `PCA()` to change the number of dimensions for reduction
- Modify `max_iter` in `LogisticRegression()` to change the maximum number of iterations for the classifier
- Change the `test_size` in `train_test_split()` to alter the train-test split ratio

## Note

A fixed random state (42) is used for reproducibility. Modify this value in `train_test_split()` and `LogisticRegression()` to explore different random initializations.

