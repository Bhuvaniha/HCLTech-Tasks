# Neural Networks From Scratch (Beginner Project)

A step-by-step, beginner-friendly project that builds a neural network using
**only Python and NumPy** — no TensorFlow, no PyTorch, no Keras, no scikit-learn.

Every file is meant to be read top-to-bottom and run on its own. The code is
deliberately kept simple and heavily commented, so you can follow the logic of
every single step.

The story is: **picture a neuron → teach it → stack neurons → push data forward
→ measure the error → push the error backward → update the weights → repeat.**
By the end, you will have trained a network that solves **XOR**, a problem a
single neuron can never solve.

---

## 1. Project folder structure

```
neural_network_from_scratch/
│
├── README.md                          <- you are here (the guide)
│
├── part01_activation_functions.py     <- Part 1: step, sigmoid, ReLU, tanh, softmax
├── part02_simple_neuron.py            <- Part 2: one neuron, step by step
├── part03_single_layer_network.py     <- Part 3: 1 neuron learns the AND gate
├── part04_forward_propagation.py      <- Part 4: 2 -> 3 -> 1 forward pass (Z1, A1, Z2, A2)
├── part05_loss_function.py            <- Part 5: Mean Squared Error
├── part06_backpropagation.py          <- Part 6: learn AND with full backprop
├── part07_mlp.py                      <- Part 7: MLP (2-4-3-1) solves XOR
├── part08_training.py                 <- Part 8: the training loop, neatly printed
├── part09_visualization.py            <- Part 9: Epoch vs Loss graph (matplotlib)
│
├── neural_network_from_scratch.py     <- Part 10: THE FINAL COMPLETE PROGRAM
│
├── loss_curve.png                     <- created by Part 9 when you run it
└── venv/                              <- Python environment you create (see below)
```

---

## 2. How to run the project

### Step A — create an environment (once)

NumPy is enough for Parts 1–8 and 10. Part 9 also needs matplotlib. To keep
things clean and avoid system-package issues, use a virtual environment:

```bash
cd neural_network_from_scratch
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install numpy matplotlib
```

### Step B — run the files, in order

```bash
python3 part01_activation_functions.py
python3 part02_simple_neuron.py
python3 part03_single_layer_network.py
python3 part04_forward_propagation.py
python3 part05_loss_function.py
python3 part06_backpropagation.py
python3 part07_mlp.py
python3 part08_training.py
python3 part09_visualization.py         # also saves loss_curve.png
python3 neural_network_from_scratch.py  # the final complete program
```

Or run the whole project at once:

```bash
for f in part0*.py neural_network_from_scratch.py; do echo "== $f =="; python3 "$f"; done
```

---

## 3. Explanation of each file

### Part 1 — `part01_activation_functions.py`  (Activation Functions)

- **WHAT:** the small formulas that turn a neuron's raw number into its output.
- **WHY:** without them the network can only learn straight lines. They add
  *non-linearity*, which is what lets a network learn XOR, images, language, etc.
  Some of them also squash the output into a fixed range (sigmoid → 0..1).
- **HOW:** each function has a plain-language comment and an example. The
  `_derivative` functions are also here, because backpropagation multiplies by
  the *slope* of the activation.
- **WHAT YOU SEE:** for inputs like `-3, -1, 0, 1, 3` you get the output of
  `step`, `sigmoid`, `relu`, `tanh`, and `softmax`, plus a note that the step
  function's slope is always 0 — which is exactly why it cannot learn.

Key idea: **activation = used forward; derivative = used backward.**

### Part 2 — `part02_simple_neuron.py`  (A Single Neuron)

- **WHAT:** the smallest building block: `inputs → weights → weighted sum + bias
  → activation → output`.
- **WHY:** a whole network is just many of these connected together.
- **HOW:** uses `inputs=[1,2]`, `weights=[0.5,0.3]`, `bias=0.1` and prints every
  step: `1*0.5 + 2*0.3 = 1.1`, then `+0.1 = 1.2` (this is **Z**), then
  `sigmoid(1.2) = 0.7685`.
- **WHAT YOU SEE:** the raw arithmetic, then the activated output, plus what
  "weight" and "bias" mean in words.

### Part 3 — `part03_single_layer_network.py`  (Single-Layer Network / Perceptron)

- **WHAT:** an input layer connected straight to **one** output neuron — the
  classic perceptron.
- **WHY:** AND is the smallest learnable problem, so it proves forward
  propagation + weight updating work.
- **HOW:** it makes a prediction, computes `error = actual - predicted` and
  nudges the weights with the perceptron rule:
  `weight = weight + learning_rate * error * input`.
- **WHAT YOU SEE:** predictions *before* training (all wrong), training progress
  per epoch, and predictions *after* (all correct) — the AND gate learned.

### Part 4 — `part04_forward_propagation.py`  (Forward Propagation)

- **WHAT:** moving data through a `2 → 3 → 1` network to produce a prediction.
- **WHY:** prediction is the first half of training. The network cannot improve
  what it cannot produce.
- **HOW:** `Z1 = X @ W1 + b1`, `A1 = sigmoid(Z1)`, `Z2 = A1 @ W2 + b2`,
  `A2 = sigmoid(Z2)`. Each value is printed.
- **Plain meaning of the letters:**
  - `Z1` = weighted sums going **into** the hidden layer (before activation)
  - `A1` = hidden layer outputs **after** activation
  - `Z2` = weighted sum going into the output neuron (before activation)
  - `A2` = final output after activation = **the prediction**
- **WHAT YOU SEE:** all intermediate values and a bonus that processes all four
  AND rows at once with matrix multiplication.

### Part 5 — `part05_loss_function.py`  (Loss / MSE)

- **WHAT:** a single number that says "how wrong is the network?".
- **WHY:** training is just a search for *smaller* loss. No measurement, no
  improvement.
- **HOW:** `MSE = average((actual - predicted)^2)` — subtract, square (so big
  errors hurt more and everything is positive), then average.
- **WHAT YOU SEE:** the loss for good predictions (~0.004), for bad predictions
  (~0.69), and for an untrained network (~0.25). Perfect predictions would give 0.

### Part 6 — `part06_backpropagation.py`  (Backpropagation)

- **WHAT:** working backwards from the error to find the gradient of every weight.
- **WHY:** this is what makes the network *learn*. Forward only predicts.
- **HOW:** the full cycle is laid out in the code:
  1. forward propagation, 2. loss, 3. error, 4. gradients (chain rule),
  5. update weights, 6. update biases, 7. repeat.
  The key formulas are written in the comments, e.g.
  `dZ2 = (2/n)*(A2 - y)*sigmoid'(Z2)`, `dW2 = A1ᵀ @ dZ2`, and
  `weight = weight - learning_rate * gradient`.
- **WHAT YOU SEE:** the network trains on AND and the loss falls from ~0.33 to
  ~0.002. Every weight change comes *only* from the gradients.

### Part 7 — `part07_mlp.py`  (Multi-Layer Perceptron / XOR)

- **WHAT:** a deeper network `2 → 4 → 3 → 1`, ReLU hidden layers, sigmoid output.
- **WHY:** XOR cannot be solved by one neuron (no straight line separates its
  classes). Stacking layers lets the network bend the decision boundary.
- **HOW:** same forward/backward/update loop as Part 6, just one extra layer.
- **WHAT YOU SEE:** predictions before training (random), the loss falling to
  ~0.00003, all four XOR answers correct — and at the bottom, a single neuron
  **failing** the same task, then a short conclusion.

> Note: XOR is famous for getting stuck in *local minima*. The file uses
> "He initialization" plus a small positive bias and a fixed seed so it trains
> reliably. If you change the seed and the loss freezes around 0.25, that is
> real behaviour — simply try another seed.

### Part 8 — `part08_training.py`  (The Training Loop)

- **WHAT:** the same XOR network, but written to highlight the loop itself.
- **WHY:** every real network (even huge ones) trains with this same
  four-step loop.
- **HOW:** `for epoch in range(...):` forward → loss → backpropagation → update.
  Loss is printed every 100 epochs.
- **WHAT YOU SEE:** output like
  ```
  Epoch 0       Loss: 0.313644
  Epoch 100     Loss: 0.023323
  ...
  Final results:
  Actual: 0 | Predicted: 0.0038
  Actual: 1 | Predicted: 0.9949
  Actual: 1 | Predicted: 0.9940
  Actual: 0 | Predicted: 0.0140
  ```

### Part 9 — `part09_visualization.py`  (Visualisation)

- **WHAT:** a graph of **Epoch vs Loss**.
- **WHY:** a picture makes learning obvious — a falling line means it is learning.
- **HOW:** trains the network, stores the loss of every epoch, then plots it.
- **WHAT YOU SEE:** a learning curve and a saved file `loss_curve.png`.
  - steep drop at the start → lots of easy improvement
  - line flattening out → learning has almost finished
  - line going up / jittering → learning rate probably too high

### Part 10 — `neural_network_from_scratch.py`  (The Final Program)

- **WHAT:** everything combined into one clean, commented program using a simple
  `NeuralNetwork` class.
- **WHY:** this is the "finished product" you can reuse and modify.
- **HOW:** the class has `forward`, `compute_loss`, `backward`,
  `update_weights`, `train`, and `predict`.
- **WHAT YOU SEE:** loss before training, loss every 200 epochs, loss after
  training, and a final `Actual | Predicted` table with an accuracy count
  (should be 4/4).

---

## 4. The complete neural-network flow

This is the whole idea in one picture. Keep it in mind while reading the files:

```
        INPUT
          │        (your data, e.g. [0, 1])
          ▼
        WEIGHTS      inputs × weights
          │
          ▼
    WEIGHTED SUM (Z)     + BIAS
          │
          ▼
    ACTIVATION FUNCTION  (sigmoid / ReLU / …)
          │
          ▼
      HIDDEN LAYER       (repeat: weighted sum → activation)
          │
          ▼
        OUTPUT            the prediction (0..1)
          │
          ▼
         LOSS             how wrong?  (MSE)
          │
          ▼
    BACKPROPAGATION       walk the error backwards, compute gradients
          │
          ▼
     WEIGHT UPDATE        weight = weight - learning_rate × gradient
          │
          ▼
        REPEAT            for many epochs, until the loss is small
```

- **Forward** = data flows left-to-right / top-to-bottom to make a prediction.
- **Backward** = the error flows right-to-left to fix the weights.
- **Training** = doing both, over and over, to make the loss smaller.

---

## 5. Things to observe while running the code

1. **Part 1:** sigmoid(0) is exactly 0.5, sigmoid_derivative(0) is 0.25 (the
   steepest slope), and the softmax outputs always add up to 1.0.
2. **Part 2:** the weighted sum (1.1) is *not* the output — the activation turns
   it into 0.7685. Also notice the bias shifts the result up.
3. **Part 3:** before training every prediction is wrong; after a handful of
   epochs every prediction is right. Watch the weights and bias change.
4. **Part 4:** `Z1` and `A1` are different — activation changes the values.
   `Z1` has 3 numbers (one per hidden neuron), `A2` has 1 (the prediction).
5. **Part 5:** the loss is small when predictions are near the true answers and
   large when they are far. Perfect predictions → 0.
6. **Part 6:** the loss drops smoothly from ~0.33 to ~0.002. It never goes
   *exactly* to 0 — gradient descent always takes finite steps.
7. **Part 7:** the MLP solves XOR while the single neuron at the bottom still
   gets rows wrong — proof that depth matters.
8. **Parts 8–10:** predictions like `0.0038` and `0.9949` are the network being
   confident and correct (close to 0 and 1).
9. **Part 9:** the loss curve falls fast, then flattens. It is *asymptotic* —
   it keeps inching towards 0 without reaching it.
10. **Experiments:** change `learning_rate` to `0.01` (slow) or `5.0` (unstable,
    loss bounces) and watch what happens. Change `epochs` and see the loss drop.

---

## 6. Five simple exercises you can try

1. **Add the OR gate.** In Part 3 (or Part 6), change the dataset to
   `OR: y = [0, 1, 1, 1]` and retrain. Does it still work? Why is OR easier
   than XOR?

2. **Make a "predict the bigger number" task.** In Part 7/10, use inputs like
   `[0.2, 0.9]`, `[0.8, 0.1]`, … and set `y = 1` when the second input is
   larger. Watch the network learn a new rule — no code structure change needed.

3. **Tune the learning rate.** Try `learning_rate = 0.01`, `0.5`, and `5.0` in
   Part 8. Plot or print the loss. Which is too slow, which is just right, and
   which is unstable?

4. **Change the architecture.** In Part 10, make the hidden layers bigger
   (`4 → 3` becomes `8 → 8`) or smaller (`2 → 2`). Does XOR still get solved?
   How many epochs does it take? This shows that architecture is a choice.

5. **Add a second output neuron (multi-output).** Extend the final program to
   predict *two* things at once, e.g. the output layer becomes size 2 and `y`
   becomes `[[0,0],[1,0],[1,0],[0,1]]`. You will learn how the same formulas
   scale to several outputs.

**Challenge (a bit harder):** add a loop that shuffles the dataset every epoch
(`np.random.permutation`) so the network sees the samples in a different order
each time. Observe whether it trains faster or more steadily (this is called
*stochastic* gradient descent).
