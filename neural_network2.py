# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

df = pd.read_csv("housing.csv")

"""dropping the target value"""

df = df.fillna(df.mean(numeric_only=True))
X = df.drop(["median_house_value", "ocean_proximity"], axis=1).values
y = df["median_house_value"].values
X = (X - X.mean(axis=0)) / X.std(axis=0)

#making the split between training set and testing set
m = len(X)
split = int(0.8 * m)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]


n_features = X_train.shape[1]
weights = np.zeros(n_features)  # weight for each feature
bias = 0                        # bias (intercept)
learning_rate = 0.01
epochs = 1000

# -------------------------
# 6️⃣ Hypothesis function
# -------------------------
def predict(x_row, bias, weights):
    return bias + np.sum(weights * x_row)

# -------------------------
# 7️⃣ Cost function (MSE)
# -------------------------
def compute_mse(X, y, bias, weights):
    errors = [predict(xi, bias, weights) - yi for xi, yi in zip(X, y)]
    mse = np.mean([e**2 for e in errors])
    return mse

# -------------------------
# 8️⃣ Gradient Descent
# -------------------------
for epoch in range(epochs):
    bias_gradient = 0
    weight_gradients = np.zeros(n_features)

    for xi, yi in zip(X_train, y_train):
        y_pred = predict(xi, bias, weights)
        error = y_pred - yi

        # Accumulate gradients
        bias_gradient += error
        weight_gradients += error * xi

    # Average gradients
    bias_gradient /= len(X_train)
    weight_gradients /= len(X_train)

    # Update parameters
    bias -= learning_rate * bias_gradient
    weights -= learning_rate * weight_gradients

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        mse = compute_mse(X_train, y_train, bias, weights)
        print(f"Epoch {epoch}: bias={bias:.2f}, MSE={mse:.2f}")
y_pred_test = [predict(xi, bias, weights) for xi in X_test]

# -------------------------
# 10️⃣ Evaluate performance
# -------------------------
mse_test = np.mean((y_pred_test - y_test)**2)
print("Test MSE:", mse_test)

class LinearRegressionScratch:
    def __init__(self, lr=0.5, n_iter=5000):
        self.lr = lr
        self.n_iter = n_iter
        self.theta = None

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)

        for _ in range(self.n_iter):
            y_pred = X.dot(self.theta)
            error = y_pred - y
            gradients = (1/m) * X.T.dot(error)
            self.theta -= self.lr * gradients

    def predict(self, X):
        return X.dot(self.theta)

r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))

print("R² Score:", r2)

# Train model
model = LinearRegressionScratch(lr=0.01, n_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation (Mean Squared Error, R²)
mse = np.mean((y_pred - y_test) ** 2)
r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))

print("MSE:", mse)
print("R² Score:", r2)

import matplotlib.pyplot as plt

y_pred = np.array([bias + np.sum(weights * xi) for xi in X_test])
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', label="Perfect fit line")
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual House Prices")
plt.show()
