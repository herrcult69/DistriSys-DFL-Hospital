# Decentralized Federated Learning Prototype

This document explains a simple decentralized federated learning (DFL) prototype for a distributed systems course project. The prototype uses three peer nodes, a small logistic regression model, and direct peer-to-peer parameter exchange without a central aggregation server.

## Project overview

The prototype simulates three hospitals. Each hospital keeps its own local training data, trains its own local model, and shares only model parameters with peers instead of sending raw data records over the network.

The current implementation uses the Breast Cancer Wisconsin dataset from scikit-learn, which is a binary classification dataset with 30 numeric features. Each node trains the same logistic regression model on a different slice of the training set, then averages weights with reachable peers.

## Why this is decentralized

In centralized federated learning, clients send updates to a central server, the server aggregates them, and then sends back the updated model. In this prototype, there is no central coordinator. Every node acts as both a learner and an aggregator by directly requesting peer parameters and computing its own average locally.

## What data is shared

The prototype does **not** share raw samples, labels, or full datasets between nodes. Each node only exposes the following information through XML-RPC (Later through we will be raw Java TCP connection):

- `node_id`: identifier of the sender node.
- `round`: current local training round.
- `w`: logistic regression weight vector.
- `b`: logistic regression bias term.[cite:132][cite:137]

For this dataset, the weight vector has one value per feature, and the scikit-learn breast cancer dataset contains 30 features. Meaning each exchange is basically a small array of floating-point parameters plus one scalar bias.

## Technologies used

| Technology | Purpose |
|---|---|
| Python 3.11 | Main implementation language and recommended runtime environment for the prototype. |
| NumPy | Vector operations and model parameter updates.|
| scikit-learn | Dataset loading, train/test split, and feature scaling.[cite:49][cite:62] |
| XML-RPC (`xmlrpc.server`, `xmlrpc.client`) | Lightweight remote procedure calls between peer nodes. |
| Threaded XML-RPC server | Allows concurrent handling of peer requests, avoiding single-thread blocking issues found in default `SimpleXMLRPCServer` usage.|
| Conda | Isolated environment setup for Python dependencies.|

## Current code structure

The implementation contains four main parts:

1. `SimpleLogisticRegression`: a minimal logistic regression model with local gradient descent training.
2. `load_local_data(node_id)`: loads the breast cancer dataset, splits it into train/test data, normalizes features, and returns one local partition for the requested node.
3. `DFLNode`: creates the local model, starts the XML-RPC server, handles peer communication, and runs round-based local training plus averaging.
4. `main`: reads command-line arguments and starts one node process.

## Dataset split

The local dataset split happens inside `load_local_data(node_id)`. The code first creates a global train/test split, then divides the training portion into three uneven partitions: 40 percent for node 1, 30 percent for node 2, and 30 percent for node 3.

This uneven split is intentional because it simulates data imbalance between hospitals, which is common in real distributed settings.The same global test set is used by every node for simple evaluation.

## Environment setup

A simple Conda environment is enough for this project. Python 3.11 is a safe choice for recent NumPy and scikit-learn versions, and scikit-learn requires Python 3.10 or newer in recent releases.[cite:62][cite:74]

Create the environment with:

```bash
conda create -n dfl-project python=3.11
conda activate dfl-project
conda install numpy scikit-learn
```

To verify the setup:

```bash
python --version
python -c "import numpy, sklearn; print('ok')"
```

## How to run the prototype

Save the code in a file such as `node.py`. Then open three terminals and start one node per terminal:

```bash
python node.py --id 1 --port 8001 --peers 8002 8003
python node.py --id 2 --port 8002 --peers 8001 8003
python node.py --id 3 --port 8003 --peers 8001 8002
```

Each process becomes one simulated hospital. It loads its local data slice, starts an XML-RPC server, trains locally for several rounds, fetches peer parameters, averages them, and prints test accuracy.

## Expected output

Each node should print:

- that the node started,
- the local dataset size,
- whether peers are reachable,
- the current training round,
- whether peer weights were fetched or skipped,
- the current test accuracy after averaging.

Since the nodes are not strictly synchronized, small timing differences between terminals are expected. This is acceptable for the current asynchronous-style prototype.

## Limitations of the current version

This prototype is intentionally simple and should be treated as a first milestone rather than a final system Important current limitations include:

- No strict synchronization barrier between nodes.
- No weighted averaging based on local dataset size.
- No secure aggregation or privacy-preserving defense against information leakage from updates.
- No node discovery protocol; peers are manually configured through ports.
- No fault-tolerant persistence or recovery if a node crashes mid-training.
- Only a basic logistic regression model is used.

