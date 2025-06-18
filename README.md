

---

````markdown
# Syntra

Syntra is a compact neural network framework that supports activation function selection, model saving/loading, early stopping, and CSV-based training. No bloated libraries, just C++ and ~Love~.

## Build

```bash
g++ -std=c++17 -O2 -o syntra syntra.cpp
````

## Usage

```bash
./syntra [flags] [optional path to CSV]
```

### Example (no CSV):

```bash
./syntra --epochs 1500 --lr 0.05 --batch 2 --activation tanh --earlystop 50 --save mymodel.txt --log mylog.txt
```

### Example (with CSV):

```bash
./syntra dataset.csv --epochs 500 --activation relu --log output.log
```

CSV rows should be in the format:

```
[input1,input2,...,output]
```

## Flags

| Flag           | Description                                              |
| -------------- | -------------------------------------------------------- |
| `--epochs`     | Number of training epochs (default: `1000`)              |
| `--lr`         | Learning rate (default: `0.1`)                           |
| `--batch`      | Mini-batch size (default: `4`)                           |
| `--activation` | Activation function: `sigmoid`, `relu`, `tanh`           |
| `--earlystop`  | Early stopping patience (default: `20`)                  |
| `--save`       | File to save trained model (default: `model.txt`)        |
| `--log`        | File to write training log (default: `training_log.txt`) |

## Outputs

* `model.txt` contains layer config and weights
* Log file shows training and validation loss per epoch

## Name?

Syntra = synapse infra. Sounds cool. Deal with it.

