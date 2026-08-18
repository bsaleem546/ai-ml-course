# Stage 4 Explained: Deep Learning with PyTorch

This walks through every task of Stage 4, in order, using one running example so the
concepts connect to something concrete instead of staying abstract: **predicting whether a
telecom customer will cancel their subscription (churn)**, using the real Telco Customer
Churn dataset this whole project has used since Stage 2. Every number below is a real result
from this project's own runs, not a made-up illustration.

The short version of what Stage 4 built: instead of using a ready-made classifier like
`LogisticRegression` (Stage 2) that fits in one line, we built a small brain-like model by
hand — deciding its shape, teaching it from examples, watching it learn, and stopping it
before it "memorized" instead of "understood."

---

## 1. Install and configure PyTorch

PyTorch is the library that does the heavy lifting: representing numbers as tensors,
computing gradients automatically, and running the training math efficiently (on CPU or
GPU). Installed with `uv add --system-certs torch`. On this machine, `torch.cuda.is_available()`
returns `False` — no GPU, so everything trains on CPU. Slower, but for a dataset this size
(~5,600 training rows), CPU is fine.

## 2. Convert a dataset into tensors

A **tensor** is just PyTorch's version of a table of numbers — like a numpy array, but one
PyTorch's math engine knows how to compute gradients through. Our raw data (customer rows —
`tenure`, `MonthlyCharges`, `Contract` type, etc.) is text and mixed types. Same preprocessing
pipeline as Stage 2 (impute missing values, scale numbers, one-hot-encode categories) turns
each customer into a row of 45 numbers. Those numbers get converted into a `torch.float32`
tensor — the only format the network can actually read. Labels (`Churn`: Yes/No) become `1`/`0`,
reshaped from a flat list into a column so they line up with what the network will output.

**Real-world framing:** think of each customer as a row on a spreadsheet — 45 columns of raw
signal about their account. The tensor conversion is just translating that spreadsheet row into
pure numbers a machine can multiply.

## 3. Implement a PyTorch Dataset

`ChurnDataset` is a small wrapper class with two methods: "how many customers do you have"
(`__len__`) and "give me customer number `i`" (`__getitem__`). This is the standard interface
PyTorch expects — once your data fits this shape, PyTorch can loop over it, shuffle it, and
batch it for you automatically.

## 4. Implement a DataLoader

Instead of showing the network all ~5,600 customers at once (too much memory, and it would
learn from one giant blurry average instead of gradually), the `DataLoader` hands out small
groups — **batches** — of customers at a time. `batch_size=32` means: show the network 32
customers, let it adjust slightly, then move to the next 32. `train_loader` shuffles customers
each pass (so the model doesn't learn any accidental ordering pattern); `val_loader` doesn't
shuffle, since validation is just measuring, not learning.

## 5–6. Build a feed-forward network + forward pass

`ChurnNet` is the actual "brain": 45 numbers in → 32 → 16 → 1 number out, with `ReLU`
(a simple "zero out anything negative" step) between each layer. Each layer is a set of
learnable weights — numbers the network adjusts during training. The **forward pass** is just
running one customer's 45 numbers through this chain of multiplications to get a single raw
number (a "logit") at the end — higher means "more likely to churn," lower means "less likely."

**Real-world framing:** imagine 32 different "junior analysts," each looking at the customer's
data and forming a rough opinion; their opinions get combined by 16 "senior analysts," who
combine into one final risk score. Except here, "opinions" are just weighted sums the network
learns from data, not human judgment.

## 7–8. Loss function + optimizer

The network's first guesses are basically random — its weights start with no idea what churn
even means. **Loss** (`BCEWithLogitsLoss`, binary cross-entropy) is a single number measuring
"how wrong was this guess" — the further the predicted risk is from the true answer (0 or 1),
the higher the loss. The **optimizer** (`Adam`, learning rate `0.001`) is the algorithm that
nudges every weight in the network slightly, in whatever direction reduces that loss.

## 9–11. Training loop, validation loop, tracking loss

Each **epoch** = one full pass through all training customers. For each batch: run the forward
pass, compute the loss, run `backward()` (PyTorch automatically computes exactly how much each
of the network's weights contributed to the error — this is "backpropagation"), then
`optimizer.step()` nudges the weights. The validation loop does the same forward pass and loss
calculation on customers the network never trains on, with no weight updates — a pure "how's it
actually doing" check.

Running 20 epochs on the real data produced a very typical shape: training loss fell smoothly
the whole time (0.49 → 0.40), while validation loss dropped quickly at first then started
creeping back up after around epoch 12–16. **This is overfitting** (Stage 3's core lesson,
showing up again here): the network kept getting better at the training customers specifically,
past the point where that was still helping it generalize to new ones.

## 12. Track task-specific metrics

Loss is useful for training but not intuitive for judging "is this actually a good churn
model." So `evaluate_metrics()` converts the network's raw output into a real prediction
(sigmoid → probability → threshold at 0.5) and reports accuracy/precision/recall/F1 — the same
metrics used since Stage 2. Result: accuracy ~0.80, recall ~0.55 — meaning the model correctly
flags a bit over half of the customers who actually go on to churn. Comparable to, but not
clearly better than, Stage 2's classical models on this same dataset — a genuinely useful
finding, not a disappointment (see the Stage 5 note below on why).

## 13–14. Experiment with batch size and learning rate

These are the two most important "dials" you can turn without changing the network's
architecture at all. Batch size sweep (`[8, 32, 128, 512]`) showed `32` performing best;
learning rate sweep (`[0.0001, 0.001, 0.01, 0.1]`) showed `0.001` performing best — with the
two extremes at each end performing clearly worse. The takeaway wasn't "we found a magic
number" — it was that the original default choices already happened to be the right ones,
confirmed with real evidence instead of assumption.

## 15. Add early stopping

If validation loss stops improving for a few epochs in a row (`patience=3`), stop training
instead of blindly running all 20 epochs — no point training further once the model's started
overfitting. In the real run, training stopped at epoch 7 (best validation loss was actually at
epoch 4) instead of running the full 20 — saving wasted computation and avoiding a worse final
model.

## 16. Save checkpoints

Every time validation loss hits a new best, save the network's current weights
(`model.state_dict()`) to disk. This means even if training runs past the best point (which it
usually will, since you don't know it's the best until later epochs prove it wasn't beaten),
you still have the actual best version saved — not just whatever the last epoch happened to
produce.

## 17. Load checkpoints for inference

Before reporting final results (or making real predictions later), reload those saved best
weights back into the model. This closes a subtle gap: without this step, "final metrics" would
describe whatever epoch training happened to stop on, not the genuinely best model you went to
the trouble of saving.

## 18. Detect CUDA and move to GPU

Defensive/portability code: check if a GPU is available (`torch.cuda.is_available()`), and if
so, move the model and every batch of data onto it. On this machine, this always resolves to
`"cpu"` — no visible behavior change — but the same code would automatically speed up training
significantly on a machine with an NVIDIA GPU, without needing to be rewritten.

## 19. Expose training as a background job in the platform

Training takes real time (even with early stopping). Instead of making an API caller wait with
the HTTP connection held open, `POST /api/v1/models/train-nn` returns instantly with a
`"queued"` job id, while the actual training runs in the background (`BackgroundTasks`).
`GET /api/v1/nn-jobs/{id}` lets you poll for progress — the same `queued → running →
completed/failed` pattern already used for CSV ingestion jobs back in Stage 1. Verified against
the real Docker stack: uploaded a dataset, kicked off training, polled until `completed`, saw
real metrics (accuracy 0.80, recall 0.54) come back, and confirmed an unknown job id correctly
returns a clean `404` instead of crashing.

## 20. Add tests for model loading and inference

Automated tests locking in two things: (1) the background job correctly does nothing when
given a job that's already finished or doesn't exist (so accidentally re-running a job twice
can't corrupt anything), and (2) the actual save → reload → forward-pass mechanics from tasks
16–17 genuinely work — save a model's weights to a temp file, load them into a brand-new model
instance, and confirm a prediction still runs correctly.

---

## Deep learning vs. "Neural Network From Scratch" (Stage 5) — what's actually different?

They're not two different technologies — **Stage 5 is Stage 4 with the training wheels taken
off, for one small example, purely to build intuition.**

"Deep learning" is the general field: using neural networks (layers of weighted connections,
like `ChurnNet`) to learn from data. Stage 4 *practiced* deep learning using PyTorch, a
framework that handles the hard math automatically — most importantly, `loss.backward()` did
something called **automatic differentiation (autograd)**: given the loss, PyTorch worked out
exactly how to adjust every one of the network's weights, using calculus, without you writing
any of that math yourself.

Stage 5 removes that framework entirely and rebuilds a tiny neural network using plain NumPy —
meaning **you** compute the forward pass, **you** compute the loss, and — the actual point of
the exercise — **you** manually work out the derivatives (gradients) and update the weights
yourself, instead of letting `.backward()` do it invisibly. It's deliberately a step backward
in capability (a tiny toy network, not something you'd deploy) in exchange for a big step
forward in understanding: once you've hand-computed a gradient once, "backpropagation" stops
being a magic black box and becomes a specific, mechanical calculus operation you did with your
own code.

So: Stage 4 = *using* deep learning, with PyTorch doing the calculus for you. Stage 5 = briefly
*being* the calculus, so that everything PyTorch did automatically in Stage 4 actually makes
sense.
