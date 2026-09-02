# Neural Network From Scratch — What Autograd Removes (Stage 5)

Manual NumPy implementation (`scripts/nn_from_scratch.py`) and PyTorch equivalent
(`scripts/nn_pytorch_comparison.py`) trained on the same tiny dataset
(`X=[1,2,3,4,5]`, `y=[2,4,5,4,5]`), same architecture (1 → 3 hidden → 1, ReLU), same
starting weights, same `SGD(lr=0.01)`, 1000 epochs. Both converged to the same loss
floor (`0.4800`) with matching final weights and the same dead ReLU unit
(`b1[1] = 0.0`) appearing independently in both runs — confirming the hand-derived
gradients below are mathematically identical to what `.backward()` computes.

## What's in the manual version that autograd eliminates

**1. The backward-pass function itself.**
`compute_gradients_multilayer()` (24 lines) doesn't exist in the PyTorch script at
all. `loss.backward()` replaces it entirely — one call computes `dW1`, `db1`, `dW2`,
`db2` together, no derivative math written by hand.

**2. Manual chain-rule propagation through each layer.**
The manual version has to explicitly walk the gradient backward one operation at a
time:

```python
d_output = -(2 / n) * (y - output)          # dLoss/doutput
dW2 = hidden.T @ d_output                    # dLoss/dW2
d_hidden = d_output @ W2.T                   # push gradient back through W2
d_z1 = d_hidden * relu_derivative(z1)        # push gradient back through ReLU
dW1 = X.T @ d_z1                             # dLoss/dW1
```

Autograd does this by building a computation graph during the *forward* pass —
every `+`, `@`, and `ReLU` call gets recorded as a graph node — then walks that
graph backward automatically. The `d_hidden = d_output @ W2.T` /
`d_z1 = d_hidden * relu_derivative(z1)` lines are literally the graph-walk, done
by hand instead of by the framework.

**3. Manually-coded derivatives for every operation.**
`relu_derivative(z)` (`(z > 0).astype(float)`) had to be derived and coded by hand.
PyTorch already knows the derivative of every built-in op (`Linear`, `ReLU`,
`MSELoss`, matrix multiply, ...) — using library ops means never writing a
derivative yourself. This only becomes a real task again if you ever build a
genuinely custom op with no existing PyTorch equivalent.

**4. Per-parameter update lines.**
The manual loop updates each parameter individually:

```python
W1 -= learning_rate * dW1
b1 -= learning_rate * db1
W2 -= learning_rate * dW2
b2 -= learning_rate * db2
```

`optimizer.step()` updates every tensor in `model.parameters()` generically — this
is the same one line whether the model has 4 parameters or 4 million; the manual
approach requires a new line per parameter as the network grows.

**5. Gradient accumulation is implicit vs. explicit.**
The manual version has no equivalent risk here — `dw`/`db` (or `dW1`/`db1`/...)
are freshly computed and overwritten every loop iteration, so there's nothing to
reset. PyTorch's autograd, by contrast, **accumulates** gradients into
`.grad` by default (adds each new `.backward()` call's gradients on top of
whatever was already there) — a deliberate design choice that supports patterns
like gradient accumulation across mini-batches or RNN backprop-through-time. This
means `optimizer.zero_grad()` is not boilerplate — forgetting it silently corrupts
training by summing gradients across epochs. Writing the manual version first
makes this a comprehensible design choice instead of a "just always call this"
rule.

## Why the manual version still matters

None of this means the manual approach is how you'd train a real network — you
wouldn't hand-write backprop for anything beyond a toy example in practice. The
value is in what it makes legible afterward:

- Debugging a real training run that produces `NaN` losses or vanishing/exploding
  gradients is far more tractable once you've built the mechanism that produces
  gradients in the first place, rather than treating `.backward()` as a black box.
- Non-standard architectures or loss functions sometimes require a custom
  `autograd.Function` with a manually-implemented backward pass — this is exactly
  `compute_gradients_multilayer`'s job, just inside PyTorch's graph instead of a
  plain Python loop.
- The shape bugs hit along the way (`y` needing `.reshape(-1, 1)` in task 7/8) are
  the same category of bug that silently produces wrong-but-plausible results in
  a framework too, except autograd won't flag a shape mismatch as "your math is
  wrong" — it'll just compute a different (still technically valid) gradient.
