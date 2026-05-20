# FAQ : Python & PyTorch for the Labs

**For students who know math but are new to PyTorch.**
Read this when you see code in the lab that confuses you.

---

## 1. What is `self`?

`self` is a reference to **the object itself**. When you create a class, `self` is how each object talks about its own data.

**Analogy:** Imagine you are filling out a form. `self.name` means "MY name." Every student fills the same form, but `self.name` is different for each one.

```python
class Student:
    def __init__(self, name):
        self.name = name       # MY name
        self.grade = 0         # MY grade

    def introduce(self):
        print(f"I am {self.name}")  # each student says their own name

alice = Student("Alice")
bob = Student("Bob")
alice.introduce()  # → "I am Alice"
bob.introduce()    # → "I am Bob"
```

**Rule:** Every method inside a class takes `self` as its first argument. Python fills it in automatically , you never write `alice.introduce(alice)`, just `alice.introduce()`.

---

## 2. What is `__init__`?

`__init__` is the **constructor** , the method that runs automatically when you create an object.

```python
class Dog:
    def __init__(self, breed):
        self.breed = breed     # this runs when you write Dog("Labrador")

rex = Dog("Labrador")         # Python calls __init__ for you
print(rex.breed)               # → "Labrador"
```

**In PyTorch**, `__init__` is where you **define the layers** of your neural network:

```python
class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()                    # see next question
        self.fc1 = nn.Linear(784, hidden)     # first layer
        self.fc2 = nn.Linear(hidden, 10)      # second layer
        self.relu = nn.ReLU()                 # activation function
```

---

## 3. What is `super().__init__()`?

When your class **inherits** from another class, `super().__init__()` calls the parent's constructor to set up its internal machinery.

**Analogy:** You are building a custom car. Before you add your custom seats and paint, you need to first build the basic chassis : engine, wheels, frame. `super().__init__()` builds the chassis.

```python
class MLP(nn.Module):           # MLP inherits from nn.Module
    def __init__(self):
        super().__init__()      # → build the nn.Module chassis first
        self.fc1 = nn.Linear(784, 256)   # → now add YOUR custom parts
```

**What happens if you forget it?** PyTorch will crash with an error like `Module.__init__() not called`. The parent class needs to register internal data structures before you add layers.

**Simple rule:** Every PyTorch model class starts with `super().__init__()` as the first line inside `__init__`.

---

## 4. What is `nn.Module`?

`nn.Module` is the **base class for all neural networks** in PyTorch. It gives you:

- automatic tracking of all parameters (weights and biases),
- `.to(device)` to move to GPU,
- `.train()` / `.eval()` to switch modes,
- `.parameters()` to get all weights for the optimizer.

**You never use `nn.Module` directly.** You always create your own class that inherits from it:

```python
class MyModel(nn.Module):       # ← your model IS an nn.Module
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)

    def forward(self, x):       # ← required: defines what happens to input
        return self.layer(x)
```

---

## 5. What is `forward`?

`forward` defines **what happens when data passes through the model**. It is the recipe.

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):           # x is the input tensor
        x = x.view(x.size(0), -1)  # flatten the image
        x = torch.relu(self.fc1(x)) # layer 1 + activation
        x = self.fc2(x)            # layer 2 (output)
        return x
```

**You never call `model.forward(x)` directly.** You call `model(x)` . PyTorch calls `forward` for you and adds extra features (hooks, gradient tracking).

```python
output = model(input_tensor)    # ✅ correct
output = model.forward(input_tensor)  # ❌ works but wrong practice
```

---

## 6. What does `.view()` or `.reshape()` do?

It changes the **shape** of a tensor without changing the data.

```python
x = torch.tensor([1, 2, 3, 4, 5, 6])   # shape: (6,)
x = x.view(2, 3)                         # shape: (2, 3) → [[1,2,3],[4,5,6]]
x = x.view(3, 2)                         # shape: (3, 2) → [[1,2],[3,4],[5,6]]
```

**In the labs**, you see:
```python
x = x.view(x.size(0), -1)    # flatten: (batch, 1, 28, 28) → (batch, 784)
```

The `-1` means "compute this dimension automatically." If batch=32 and total elements=32×784, then `-1` becomes 784.

---

## 7. What is a tensor?

A tensor is a **multi-dimensional array** . It is like a numpy array but with GPU support and automatic gradients.

| Math name | Dimensions | Python shape | Example |
|-----------|-----------|--------------|---------|
| scalar | 0 | `()` | `torch.tensor(3.14)` |
| vector | 1 | `(n,)` | `torch.tensor([1, 2, 3])` |
| matrix | 2 | `(m, n)` | pixel values of one image |
| 3D tensor | 3 | `(batch, height, width)` | batch of grayscale images |
| 4D tensor | 4 | `(batch, channels, H, W)` | batch of color images |

**In the MNIST lab:**
```python
images.shape   # → torch.Size([128, 1, 28, 28])
#                    batch=128, channels=1, height=28, width=28
```

---

## 8. What is `.to(device)`?

It moves a tensor or model to CPU or GPU.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MLP().to(device)       # move all weights to GPU
x = x.to(device)               # move input data to GPU
```

**Rule:** The model AND the data must be on the **same device**. If the model is on GPU and the data is on CPU, PyTorch crashes.

---

## 9. What is `model.train()` vs `model.eval()`?

| Mode | What changes | When to use |
|------|-------------|-------------|
| `model.train()` | Dropout is active, BatchNorm uses batch stats | during training |
| `model.eval()` | Dropout is off, BatchNorm uses running stats | during evaluation/testing |

```python
model.train()          # before training loop
for x, y in train_loader:
    ...

model.eval()           # before testing
with torch.no_grad():  # also disable gradient computation (saves memory)
    for x, y in test_loader:
        ...
```

---

## 10. What is `torch.no_grad()`?

It tells PyTorch: **do not compute gradients**. Saves memory and speeds up evaluation.

```python
# During training : gradients are needed
output = model(x)
loss = criterion(output, y)
loss.backward()         # ← computes gradients

# During testing : no gradients needed
with torch.no_grad():   # ← saves memory
    output = model(x)
    acc = (output.argmax(1) == y).float().mean()
```

---

## 11. What is `loss.backward()`?

It computes the **gradient of the loss with respect to every parameter**. This is backpropagation.

```python
loss = criterion(model(x), y)  # compute loss
loss.backward()                 # compute all gradients (∂loss/∂w for every w)
optimizer.step()                # update weights using those gradients
optimizer.zero_grad()           # reset gradients for next iteration
```

**The 4-step training loop** . Memorize this pattern:
```python
for x, y in train_loader:
    output = model(x)           # 1. forward pass
    loss = criterion(output, y) # 2. compute loss
    loss.backward()             # 3. compute gradients
    optimizer.step()            # 4. update weights
    optimizer.zero_grad()       # (reset for next batch)
```

---

## 12. What is `nn.CrossEntropyLoss()`?

The loss function for **classification**. It combines softmax + negative log-likelihood in one step.

**Input:** raw scores (logits) from the model , NOT probabilities.
**Target:** class indices (0, 1, 2, ... 9 for MNIST).

```python
criterion = nn.CrossEntropyLoss()
logits = model(x)              # shape: (batch, 10) : raw scores
loss = criterion(logits, y)    # y is shape: (batch,) : integers 0-9
```

**Common mistake:** Do NOT apply softmax before CrossEntropyLoss. It already does it internally.

---

## 13. What is `nn.Linear(in, out)`?

A **fully connected layer**: `output = input × W + b`

```python
layer = nn.Linear(784, 256)
# W has shape (256, 784) : 200,704 parameters
# b has shape (256,) : 256 parameters
# Total: 200,960 parameters

x = torch.randn(32, 784)      # 32 samples, 784 features each
output = layer(x)              # shape: (32, 256)
```

---

## 14. What is `nn.Conv2d(in_channels, out_channels, kernel_size)`?

A **convolution layer** for images. Slides a small filter across the image.

```python
conv = nn.Conv2d(1, 32, 3, padding=1)
# Input: 1 channel (grayscale), Output: 32 feature maps, Filter: 3×3
# padding=1 keeps the spatial size unchanged

x = torch.randn(32, 1, 28, 28)   # batch of 32 grayscale images
output = conv(x)                   # shape: (32, 32, 28, 28)
```

---

## 15. What is `nn.LSTM(input_size, hidden_size)`?

A **recurrent layer** with memory gates. Processes sequences step by step.

```python
lstm = nn.LSTM(input_size=32, hidden_size=128, batch_first=True)

x = torch.randn(16, 28, 32)       # batch=16, sequence_length=28, features=32
output, (h_n, c_n) = lstm(x)

# output: (16, 28, 128) : hidden state at EVERY time step
# h_n:    (1, 16, 128)  : hidden state at the LAST time step
# c_n:    (1, 16, 128)  : cell state at the LAST time step
```

**In the MNIST lab**, we treat each row of the image (28 pixels) as one time step. So 28 rows = 28 time steps.

---

## 16. What does `squeeze()` and `unsqueeze()` do?

They **add or remove dimensions of size 1**.

```python
x = torch.randn(32, 1, 28, 28)

x.squeeze(1)      # remove dim 1 → shape (32, 28, 28)
x.squeeze()       # remove ALL dims of size 1

x = torch.randn(32, 128)
x.unsqueeze(0)    # add dim at position 0 → shape (1, 32, 128)
x.unsqueeze(-1)   # add dim at end → shape (32, 128, 1)
```

---

## 17. What is `DataLoader`?

It takes a dataset and gives you **mini-batches** for training.

```python
loader = DataLoader(dataset, batch_size=128, shuffle=True)

for batch_x, batch_y in loader:
    # batch_x shape: (128, 1, 28, 28)
    # batch_y shape: (128,)
    ...
```

- `batch_size=128`: process 128 samples at once
- `shuffle=True`: randomize order every epoch (important for training)

---

## 18. What is `.detach()` and why do I need it?

`.detach()` **cuts the gradient connection**. Use it when you want to use a tensor's value without tracking gradients.

```python
# During attention visualization:
self.attn_weights = weights.detach()   # save weights without gradient tracking

# Converting to numpy (numpy does not support gradients):
values = tensor.detach().cpu().numpy()  # detach → move to CPU → convert
```

---

## 19. Why `optimizer.zero_grad()` : why reset gradients?

PyTorch **accumulates gradients by default**. If you do not reset, the gradient from batch 1 is added to the gradient from batch 2, and you get wrong updates.

```python
# Without zero_grad (WRONG):
# batch 1: gradient = 0.5
# batch 2: gradient = 0.5 + 0.3 = 0.8  ← accumulated, wrong!

# With zero_grad (CORRECT):
optimizer.zero_grad()   # reset to 0
loss.backward()         # gradient = 0.3  ← correct
optimizer.step()
```

---

## 20. Quick reference : shapes in the MNIST lab

| Stage | Shape | Explanation |
|-------|-------|-------------|
| Raw image | `(1, 28, 28)` | 1 channel, 28×28 pixels |
| Batch of images | `(128, 1, 28, 28)` | 128 images |
| After flatten | `(128, 784)` | 28×28 = 784 |
| After `nn.Linear(784, 256)` | `(128, 256)` | 256 hidden features |
| After `nn.Linear(256, 10)` | `(128, 10)` | 10 class scores |
| Labels | `(128,)` | integers 0–9 |
| After `argmax(1)` | `(128,)` | predicted class per sample |

---

*AIMS Senegal | Applied Generative & Agentic AI | Dr. Papa-Séga WADE*
