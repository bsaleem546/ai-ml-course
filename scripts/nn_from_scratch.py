import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=np.float64)   # hours studied
y = np.array([2, 4, 5, 4, 5], dtype=np.float64)    # exam score (out of 10, made up but plausible)

X_col = X.reshape(-1, 1)  # shape (5, 1) — 5 examples, 1 feature each

np.random.seed(42)
hidden_size = 3

W1 = np.random.randn(1, hidden_size) * 0.1   # input -> hidden weights
b1 = np.zeros(hidden_size)                    # hidden biases
W2 = np.random.randn(hidden_size, 1) * 0.1    # hidden -> output weights
b2 = np.zeros(1)                              # output bias

w = 0.0
b = 0.0

def forward(X, w, b):
    return w * X + b

predictions = forward(X, w, b)
# print(predictions)

def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# loss = compute_loss(y, predictions)
# # print(loss)

def compute_gradients(X, y_true, y_pred):
    n = len(X)
    dw = -(2 / n) * np.sum(X * (y_true - y_pred))
    db = -(2 / n) * np.sum(y_true - y_pred)
    return dw, db

# dw, db = compute_gradients(X, y, predictions)
# # print(dw, db)

# learning_rate = 0.01
# w = w - learning_rate * dw
# b = b - learning_rate * db
# print(w, b)

def relu(x):
    return np.maximum(0, x)

def forward_multilayer(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1        # pre-activation
    hidden = relu(z1)       # post-activation
    output = hidden @ W2 + b2
    return z1, hidden, output

z1, hidden, output = forward_multilayer(X_col, W1, b1, W2, b2)
print("z1 shape:", z1.shape)
print("hidden shape:", hidden.shape)
print("output shape:", output.shape)
print(output)



# epochs = 1000
# learning_rate = 0.01

# for epoch in range(epochs):
#     predictions = forward(X, w, b)
#     loss = compute_loss(y, predictions)
#     dw, db = compute_gradients(X, y, predictions)
#     w = w - learning_rate * dw
#     b = b - learning_rate * db

#     if epoch % 100 == 0:
#         print(f"epoch {epoch}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

# print(f"final: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")