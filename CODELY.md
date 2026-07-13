# CODELY.md

## Project Overview

This is a Python 3 port of the code samples accompanying Michael Nielsen's book
["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com).
The original code was written for Python 2.6/2.7; this fork targets Python 3.8–3.10.

The project implements feedforward neural networks from scratch using only NumPy
(for `network.py` and `network2.py`), and Theano (for `network3.py` and `conv.py`).
It classifies the MNIST handwritten digit dataset (28×28 pixel images, 784 features).

### Key Technologies

- **Python 3.10** (managed via [uv](https://github.com/astral-sh/uv) v0.11.28)
- **NumPy** (< 1.22) — matrix operations, backpropagation
- **Theano** — symbolic computation, GPU support for `network3.py`/`conv.py`
- **scikit-learn** (< 1.1) — SVM baseline classifier
- **matplotlib** (3.x) — visualization scripts in `fig/`

### Architecture

The code progresses in complexity across book chapters:

| File | Chapter | Description |
|---|---|---|
| `src/network.py` | 1 | Basic feedforward NN: SGD + backprop, quadratic cost, sigmoid activations |
| `src/network2.py` | 3 | Improved NN: cross-entropy cost, L2 regularization, better weight init, save/load |
| `src/network3.py` | 6 | Theano-based NN: fully-connected, conv+pool, softmax layers, dropout, GPU support |
| `src/conv.py` | 6 | Convolutional network experiments (shallow, basic conv, double conv, ReLU, dropout, ensemble) |
| `src/mnist_loader.py` | — | Loads MNIST from `data/mnist.pkl.gz`; provides raw and wrapper formats |
| `src/expand_mnist.py` | — | Data augmentation: shifts each training image ±1px → 250k images |
| `src/mnist_average_darkness.py` | — | Naive baseline classifier (average pixel darkness per digit) |
| `src/mnist_svm.py` | — | SVM baseline classifier using scikit-learn |

## Building and Running

### Environment Setup

```bash
# Create and activate the virtual environment
uv venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Scripts

**Important:** Scripts use relative paths (`../data/`) and `sys.path.append('../src/')`,
so they must be run from specific directories:

```bash
# Run from src/ directory — uses ../data/mnist.pkl.gz
cd src
python network.py        # No __main__; import and use interactively
python mnist_svm.py      # Has __main__; runs SVM baseline
python mnist_average_darkness.py  # Has __main__; runs darkness baseline
python expand_mnist.py   # Has __main__; generates ../data/mnist_expanded.pkl.gz

# Run from fig/ directory — imports from ../src/
cd fig
python overfitting.py    # Interactive prompts for parameters
python more_data.py      # Runs NN + SVM comparison across training set sizes
```

### Typical Usage (Interactive)

```python
# From src/ directory
import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```

### No Formal Test Suite

This project has no automated tests. Verification is done by running the scripts
and checking printed accuracy / generated plots.

### No Build System

Pure Python scripts — no compilation, packaging, or build step required.

## Development Conventions

### Coding Style

- **Educational focus:** Code prioritizes readability and clarity over performance
  or completeness. Comments and docstrings explain the "why" alongside the "what."
- **Docstrings:** Every class and public method has a docstring (triple-quoted `"""`).
- **Module-level docstrings:** Each file begins with a `"""module_name ~~~~ ..."""` block.
- **Naming:** `snake_case` for functions/methods/variables, `PascalCase` for classes.
- **Section headers:** `####` used for major section breaks within files.
- **Imports:** Grouped as Standard library → Third-party → Local (`import mnist_loader`).

### Data Flow

- `mnist_loader.load_data()` returns raw tuples `(training_data, validation_data, test_data)`
  where each is `(ndarray_of_images, ndarray_of_labels)`.
- `mnist_loader.load_data_wrapper()` returns lists of `(x, y)` tuples reshaped for
  the neural network code: `x` is `(784, 1)`, `y` is `(10, 1)` one-hot vector for
  training data, or an integer for validation/test data.
- `network3.py` uses Theano shared variables for GPU compatibility
  (`load_data_shared()`).

### File Organization

- `src/` — All neural network and classifier source code.
- `data/` — MNIST dataset (`mnist.pkl.gz`) and generated expanded data.
- `fig/` — Visualization scripts (`.py`) and their outputs (`.png`, `.json`).
  Scripts in `fig/` import from `src/` via `sys.path.append('../src/')`.

### Saved Networks

`network2.py` supports saving/loading trained networks as JSON files
(`Network.save(filename)` / `network2.load(filename)`).

### License

MIT License (see `README.md`).
