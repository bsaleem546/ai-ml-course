import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=np.float64)   # hours studied
y = np.array([2, 4, 5, 4, 5], dtype=np.float64).reshape(-1, 1)   # exam score (out of 10, made up but plausible)

X_col = X.reshape(-1, 1)  # shape (5, 1) — 5 examples, 1 feature each

np.random.seed(42)
hidden_size = 3

W1 = np.random.randn(1, hidden_size) * 0.1   # input -> hidden weights
b1 = np.zeros(hidden_size)                    # hidden biases
W2 = np.random.randn(hidden_size, 1) * 0.1    # hidden -> output weights
b2 = np.zeros(1)                              # output bias

def forward(X, w, b):
    return w * X + b

def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_gradients(X, y_true, y_pred):
    n = len(X)
    dw = -(2 / n) * np.sum(X * (y_true - y_pred))
    db = -(2 / n) * np.sum(y_true - y_pred)
    return dw, db


def relu(x):
    return np.maximum(0, x)

def relu_derivative(z):
    return (z > 0).astype(np.float64)

def forward_multilayer(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    hidden = relu(z1)
    output = hidden @ W2 + b2
    return z1, hidden, output

def compute_gradients_multilayer(X, y, z1, hidden, output, W2):
    n = X.shape[0]
    d_output = -(2 / n) * (y - output)          # (5, 1)
    dW2 = hidden.T @ d_output                    # (3, 1)
    db2 = d_output.sum(axis=0)                   # (1,)
    d_hidden = d_output @ W2.T                   # (5, 3)
    d_z1 = d_hidden * relu_derivative(z1)        # (5, 3)
    dW1 = X.T @ d_z1                             # (1, 3)
    db1 = d_z1.sum(axis=0)                       # (3,)
    return dW1, db1, dW2, db2

epochs = 1000
learning_rate = 0.01

for epoch in range(epochs):
    z1, hidden, output = forward_multilayer(X_col, W1, b1, W2, b2)
    loss = compute_loss(y, output)
    dW1, db1, dW2, db2 = compute_gradients_multilayer(X_col, y, z1, hidden, output, W2)
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 100 == 0:
        print(f"epoch {epoch}: loss={loss:.4f}")
        
    if epoch == epochs - 1:
        print(f"final: loss={loss:.4f}")
        print("W1:", W1)
        print("b1:", b1)
        print("W2:", W2)
        print("b2:", b2)
