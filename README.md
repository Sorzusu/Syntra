# Syntra

Syntra is a lightweight neural network framework built in C++ with support for model saving, training from CSV datasets, multiple activation functions, and early stopping.
**It is basically zeus nn V2**

## Features

- Feedforward neural networks
- Activation functions: Sigmoid, ReLU, Tanh
- Model saving/loading to `.txt`
- CSV-based dataset input (optional)
- Training logs to file
- Early stopping

## Build

```bash
g++ -std=c++17 -O2 -o syntra main.cpp
```

## Usage

```bash
./syntra [csv_file.csv] [flags]
```

### Flags

| Flag           | Description                     |
|----------------|---------------------------------|
| `--epochs`     | Training epochs (default: 1000) |
| `--lr`         | Learning rate (default: 0.1)    |
| `--batch`      | Batch size (default: 4)         |
| `--activation` | `sigmoid`, `relu`, or `tanh`    |
| `--earlystop`  | Patience in epochs (default: 20)|
| `--save`       | Output model file               |
| `--log`        | Output log file                 |

### Example

```bash
./syntra --epochs 1500 --lr 0.05 --batch 2 --activation tanh --earlystop 50 --save model.txt --log out.log
```

### Example with CSV

```bash
./syntra dataset.csv --epochs 500 --lr 0.01 --activation relu
```

> Each CSV row should contain input values followed by output labels.

---

Licensed under the BSD 3-Clause License.
