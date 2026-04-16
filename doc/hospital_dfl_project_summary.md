# Peer-to-Peer Hospital Learning Network

This document presents a complete summary of the distributed systems course project: a hospital-themed decentralized federated learning simulation that begins with a simple Python proof of concept and evolves into a more comprehensive Java implementation with real TCP-based peer-to-peer communication.[cite:26][cite:141][cite:143]

## Project theme

The theme of the project is a network of simulated hospitals that want to collaborate on predictive medicine without pooling their raw patient data into one central database.[cite:26][cite:128][cite:133] Each hospital trains a local prediction model on its own data partition and only exchanges model parameters with peers, which matches the main idea of federated learning where data stays local while models are updated collaboratively.[cite:128][cite:132][cite:133]

The hospital setting is important because medical data is sensitive and naturally motivates a decentralized design. For the course project, the hospital scenario gives a realistic story while still allowing the implementation to stay small enough to finish in one semester project timeline.[cite:26][cite:2]

## Main project idea

The core idea is to simulate 3 to 5 hospitals as separate nodes. Each node trains a local machine learning model, then periodically exchanges model parameters such as weights and bias with neighboring hospitals through a peer-to-peer network.[cite:26][cite:141][cite:147]

Unlike centralized federated learning, there is no global coordination server that collects all updates and redistributes a single global model. Instead, every hospital is both a learner and a communication peer, which makes the system decentralized.[cite:139][cite:144][cite:152]

## Why start with Python

The project starts with a basic Python script because Python makes it easy to understand the learning logic first. The starter prototype uses NumPy and scikit-learn's built-in Breast Cancer Wisconsin dataset, which is a binary classification dataset with 30 numeric features and is easy to use for early experiments.[cite:49][cite:62]

The Python version is not meant to be the final submission architecture. Its purpose is to prove that the learning concept works: split data into local hospital partitions, train locally, exchange only model parameters, average them, and observe that a collaborative process can be implemented without sharing raw data.[cite:26][cite:27][cite:128]

## Python prototype summary

The first working prototype includes a small custom logistic regression model implemented with a weight vector and a bias term.[cite:42][cite:43] The model trains locally using gradient descent on each node's local data slice, then shares only its current parameters with peer nodes.[cite:42][cite:43][cite:132]

The prototype initially used XML-RPC because it was a convenient way to simulate network calls between nodes. XML-RPC made it easy to fetch the latest parameters from peers, but it also hid much of the low-level networking behavior and was not ideal for a full distributed systems implementation.[cite:104][cite:125][cite:154]

## What the Python prototype actually shares

The Python node does not send local training samples, labels, or full datasets across the network.[cite:128][cite:133] Instead, it sends a small model update package containing:

- node id,
- current round number,
- weight vector,
- bias term.[cite:132][cite:137]

Because the starter dataset has 30 features, the shared parameter vector is simply a list of 30 floating-point values plus one bias scalar.[cite:49] This is enough for another node to merge the received model with its own using plain or weighted averaging.[cite:149][cite:132]

## Why move to Java

The final project should move to Java because the course emphasis is on distributed systems, networking, concurrency, and explicit peer communication. Java is a strong fit for socket programming, multithreaded request handling, and process-based node simulation, and it also matches the user's prior study of Java TCP networking and concurrency patterns.[cite:154][cite:155][cite:26]

The Java rebuild makes the project feel more like a true distributed systems implementation because communication is no longer hidden behind XML-RPC. Instead, the team defines the transport protocol, message format, connection logic, threading model, and synchronization strategy directly.[cite:177][cite:183][cite:2]

## Final Java direction

The final Java implementation should use raw TCP sockets instead of XML-RPC. Each hospital node should open a `ServerSocket` to listen for incoming peer connections and use `Socket` connections to send requests or updates to neighbors.[cite:175][cite:177][cite:183]

Each node will behave as both a server and a client. That means a node must be able to accept incoming requests from peers while also initiating outgoing requests to ask for peer model parameters.[cite:159][cite:160][cite:177]

## Technologies for the Java implementation

| Area | Technology | Role |
|---|---|---|
| Language | Java | Main implementation language for the final distributed system.[cite:154] |
| Networking | `ServerSocket`, `Socket` | Peer-to-peer TCP communication between hospital nodes.[cite:175][cite:177][cite:183] |
| Concurrency | `Thread`, `ExecutorService`, `BlockingQueue`, synchronization/locks | Manage training, listening, peer requests, and safe shared model access.[cite:155][cite:220][cite:223] |
| Message format | JSON over TCP or simple line-based protocol | Encodes round number, node id, model weights, and metadata.[cite:179][cite:182] |
| Build tool | Maven or Gradle | Dependency management and project organization. |
| ML baseline | Custom logistic regression in Java | Simple local learner with transparent implementation.[cite:170] |
| Optional ML library | Tribuo | Java-native library for later experiments with classification.[cite:162][cite:195] |
| Optional advanced library | Deeplearning4j | Optional for more complex models later, if needed.[cite:164][cite:161] |

## Concepts the Java version should demonstrate

The Java version should clearly show the following distributed systems concepts:

- decentralized architecture, because there is no central aggregation server,[cite:141][cite:143]
- peer-to-peer communication, because each node talks directly to other hospitals,[cite:147][cite:177]
- concurrency, because training and networking happen in parallel,[cite:155][cite:159]
- synchronization, because multiple threads may read and update the same model state,[cite:155]
- fault tolerance basics, because peers may be offline, delayed, or unreachable,[cite:188][cite:160]
- protocol design, because the team defines the application-level message format,[cite:183][cite:179]
- data heterogeneity, because each hospital has a different local partition.[cite:26][cite:221]

These concepts are academically central to the project and are more important than using a highly advanced machine learning model.[cite:2][cite:26]

## Data and model choice

The simplest starting point is still logistic regression, because it is easy to implement by hand and easy to explain in a report or presentation.[cite:42][cite:170] The earliest dataset should remain the Breast Cancer Wisconsin dataset, since it is easy to load and useful for a binary classification baseline.[cite:49]

Once the distributed communication layer is stable, the project can move to a more hospital-like dataset such as hospital readmission data. That later upgrade strengthens the medical theme, but it should not come before the networking and decentralized protocol are working reliably.[cite:19][cite:25][cite:26]

## Rounds flow

A round is the main repeated cycle in the system. Every hospital node performs local training, exchanges model information, aggregates the received updates, evaluates the merged model, and then begins the next round.[cite:28][cite:205]

A simple round flow is:

1. Load current local model.
2. Train locally on the hospital's private dataset for a few epochs.[cite:28][cite:203]
3. Contact peer hospitals over TCP and request their latest model parameters.[cite:175][cite:183]
4. Receive peer replies containing weights, bias, round number, and optionally sample count.[cite:179][cite:171]
5. Aggregate local and peer parameters using simple or weighted averaging.[cite:149][cite:28]
6. Replace the old local model with the merged model.
7. Evaluate and log metrics such as round number, received peer count, and accuracy.
8. Start the next round.[cite:204][cite:205]

## Input and output of one round

| Stage | Input | Output |
|---|---|---|
| Local training | Current model, local dataset, learning rate, epochs | Updated local weights and bias.[cite:28][cite:42] |
| Peer request | Peer addresses, current round, request message | Outgoing TCP request.[cite:175][cite:183] |
| Peer response | Request from another node | Sender node id, round, weights, bias, optional sample count.[cite:179][cite:171] |
| Aggregation | Local model plus received peer models | New merged local model.[cite:149][cite:205] |
| Evaluation | Merged model and test data | Accuracy/loss logs for the current round.[cite:204] |

## Suggested communication protocol

A clean first protocol can use JSON messages over TCP. For example, a node could send a request like:

```json
{"type":"GET_WEIGHTS","fromNode":1,"round":3}
```

A reply could look like:

```json
{"type":"WEIGHTS","nodeId":2,"round":3,"sampleCount":136,"bias":0.13,"weights":[0.2,-0.1,0.05]}
```

This is simple enough to debug manually and strong enough for a course project report.[cite:179][cite:182]

## Suggested Java architecture

A clean codebase can be split into the following classes:

- `NodeMain`: starts one hospital node process.
- `NodeServer`: listens for incoming TCP connections.
- `PeerClient`: opens connections to peers and sends requests.
- `ConnectionHandler`: handles one inbound request in its own thread or executor task.
- `Message` and `MessageParser`: encode and decode JSON or text protocol messages.
- `ModelParameters`: stores weights, bias, round, and sample count.
- `LogisticRegressionModel`: trains locally and predicts labels.[cite:170]
- `Aggregator`: implements simple average, weighted average, or future methods.[cite:149]
- `DataLoader`: loads data and splits it by node.
- `ExperimentRunner`: controls rounds, timing, and evaluation.
- `Config`: stores node id, peers, ports, topology, and training settings.

This keeps the networking layer and ML layer separate, which makes team development easier.[cite:26][cite:154]

## Suggested team distribution

A 5-person team can split the project into two broad workstreams: machine learning and distributed systems.[cite:26] One or two members can focus on implementing and experimenting with the local learning layer, while the remaining members focus on peer-to-peer networking, concurrency, and system behavior.[cite:154][cite:155]

A practical distribution is shown below.

| Team role | Main responsibilities |
|---|---|
| Member 1: ML baseline | Implement logistic regression in Java, local training loop, prediction, feature normalization.[cite:170] |
| Member 2: ML experiments | Try new datasets, test weighted averaging and alternatives to plain average, run experiments and compare methods.[cite:149][cite:19][cite:25] |
| Member 3: TCP networking | Build `ServerSocket`/`Socket` communication, request/reply flow, peer startup logic.[cite:175][cite:177] |
| Member 4: Concurrency and coordination | Handle multi-threading, executor pools, queues, synchronization, and peer timeouts.[cite:155][cite:220][cite:223] |
| Member 5: Evaluation and integration | Metrics, logging, experiment runner, isolated-vs-collaborative comparison, integration testing.[cite:22][cite:26] |

## What each subgroup should focus on

### Machine learning subgroup

The ML subgroup should first create a stable baseline model in Java. The first milestone is not high accuracy but correctness: local training should work, parameters should update properly, and the model should expose weights in a format that can be shared over the network.[cite:170][cite:26]

After that baseline works, the subgroup can explore:

- different datasets,
- weighted averaging by sample count,
- simple neighbor-only aggregation,
- asynchronous update usage,
- eventually a more advanced model only if the team has time.[cite:149][cite:207][cite:224]

### Distributed systems subgroup

The distributed systems subgroup should build the real decentralized behavior. The main goal is to create hospital nodes that can join, listen, connect to peers, exchange model messages, and continue operating even when some peers respond late or fail.[cite:154][cite:160]

The first milestone for this subgroup is a successful TCP message exchange between nodes. After that, they should add multithreading, timeout handling, better logging, queue-based request handling if needed, and topology-aware peer communication.[cite:159][cite:220][cite:223]

## Development plan from simple to final

A realistic execution path is:

### Phase 1: Python proof of concept

- Implement a basic local logistic regression prototype.
- Split one simple dataset into 3 hospital partitions.
- Share only parameters, not raw data.
- Confirm that multi-round averaging works.[cite:42][cite:49][cite:132]

### Phase 2: Java baseline

- Recreate logistic regression in Java.
- Recreate dataset loading and local training.
- Run one node locally without networking.[cite:170]

### Phase 3: Java networking

- Start separate Java node processes.
- Implement TCP request/reply communication.
- Exchange model parameters manually between nodes.[cite:175][cite:183]

### Phase 4: Java decentralized rounds

- Add round control and peer requests.
- Merge local and peer models after each round.
- Log round metrics and peer status.[cite:205][cite:149]

### Phase 5: Robustness and evaluation

- Add timeouts, skipped-peer handling, and retry policy.[cite:188][cite:160]
- Compare isolated local training with collaborative decentralized training.[cite:22][cite:26]
- Test alternative aggregation methods and possibly new datasets.[cite:149][cite:19]

## What the final project should achieve

By the end, the final project should demonstrate that multiple hospital nodes can train on private local data, exchange only compact model updates over TCP, and improve or at least meaningfully compare collaborative learning against isolated local learning without any central coordination server.[cite:26][cite:141][cite:143]

A successful final system does not need to be industrial-scale. It only needs to be technically coherent, clearly decentralized, experimentally evaluated, and well explained.[cite:2][cite:26]

## Recommended final description

A concise final description of the project is:

> A peer-to-peer hospital simulation in which multiple Java nodes train local predictive models on private data partitions and exchange model parameters over TCP sockets in repeated decentralized learning rounds, without using any central aggregation server.[cite:141][cite:143][cite:175]

This description is accurate, academically strong, and understandable to someone reading the proposal or report for the first time.[cite:2][cite:26]
