import sklearn.datasets as skdata
import sklearn.decomposition as skdecomp
import matplotlib.pyplot as plt
import sklearn.model_selection as skmodel
import sklearn.linear_model as sklinear
import sklearn.metrics as skmetrics
import numpy as np

# Retrieve cancer dataset
tumor_data = skdata.load_breast_cancer()
X = tumor_data.data
y = tumor_data.target

# Perform PCA
pca_transformer = skdecomp.PCA()
X_transformed = pca_transformer.fit_transform(X)

# Output variance ratios
print("Component Variance Ratios:")
print(pca_transformer.explained_variance_ratio_)

# Plot cumulative variance
cum_var = np.cumsum(pca_transformer.explained_variance_ratio_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var, 'r.-')
plt.xlabel('Principal Component Count')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance Explanation vs. Component Count')
plt.tight_layout()
plt.show()

# Reduce to 2D
pca_2d = skdecomp.PCA(n_components=2)
X_2d = pca_2d.fit_transform(X)

print(f"\nShape after dimension reduction: {X_2d.shape}")

# Split dataset
X_train, X_test, y_train, y_test = skmodel.train_test_split(X_2d, y, test_size=0.2, random_state=42)

# Initialize and fit logistic regression
classifier = sklinear.LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Compute accuracy
acc = skmetrics.accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {acc:.4f}")
