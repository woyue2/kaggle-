
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#手撕实现逻辑回归（不使用机器学习的库）
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_descent(X, y, alpha, epochs):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    for _ in range(epochs):
        y_pred = sigmoid(np.dot(X, theta))
        loss = -(1/m) * np.sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))
        gradient = (1 / m) * np.dot(X.T, y_pred - y)
        theta -= alpha * gradient
        losses.append(loss)
    return theta, losses

def logistic_regression(X, y, alpha=0.01, epochs=1000):
    m, n = X.shape
    X = np.column_stack((np.ones(m), X))  
    y = y.astype(float)
    theta, losses = gradient_descent(X, y, alpha, epochs)
    return theta, losses


data = pd.read_csv('breast-cancer-wisconsin.data', header=None)
data = data.replace('?', np.nan).dropna().astype(int)  
target = data.iloc[:, -1].replace({2: 0, 4: 1}).values  
features = data.iloc[:, 1:-1].values


np.random.seed(42)
indices = np.random.permutation(len(target))
train_size = int(0.8 * len(target))
train_indices = indices[:train_size]
test_indices = indices[train_size:]
X_train, y_train = features[train_indices], target[train_indices]
X_test, y_test = features[test_indices], target[test_indices]


theta, losses = logistic_regression(X_train, y_train)
X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))  
y_pred = (sigmoid(np.dot(X_test, theta)) > 0.5).astype(int)  


correct_predictions = np.sum(y_test == y_pred)
accuracy = correct_predictions / len(y_test)
print('Accuracy:', accuracy)

plt.plot(range(len(losses)), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
