import torch
import torch.nn as nn

# Same tiny dataset as scripts/nn_from_scratch.py
X = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).reshape(-1, 1)
y = torch.tensor([2, 4, 5, 4, 5], dtype=torch.float32).reshape(-1, 1)

# Same starting weights as nn_from_scratch.py (np.random.seed(42), hidden_size=3),
# so this is a fair comparison of training methods, not initialization.
W1 = [[0.04967141530112327, -0.013826430117118467, 0.06476885381006925]]
b1 = [0.0, 0.0, 0.0]
W2 = [[0.15230298564080255], [-0.023415337472333597], [-0.023413695694918055]]
b2 = [0.0]

model = nn.Sequential(
    nn.Linear(1, 3),
    nn.ReLU(),
    nn.Linear(3, 1),
)

with torch.no_grad():
    # nn.Linear stores weight as (out_features, in_features), NumPy's W1 was
    # (in_features, out_features) = (1, 3), hence the transpose.
    model[0].weight.copy_(torch.tensor(W1).T)
    model[0].bias.copy_(torch.tensor(b1))
    model[2].weight.copy_(torch.tensor(W2).T)
    model[2].bias.copy_(torch.tensor(b2))

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    output = model(X)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch {epoch}: loss={loss.item():.4f}")

    if epoch == epochs - 1:
        print(f"final: loss={loss.item():.4f}")
        print("W1:", model[0].weight.data)
        print("b1:", model[0].bias.data)
        print("W2:", model[2].weight.data)
        print("b2:", model[2].bias.data)
