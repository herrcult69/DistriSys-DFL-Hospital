# Distributed Federated Learning — Hospital Symptom Checker

A 3-node federated learning system where simulated hospitals collaboratively train
a GPT-2 LoRA language model on medical diagnostic data without sharing raw patient records.

Each node runs a **Java coordinator** (P2P networking, round orchestration) and a
**Python ML server** (local training, FedIT aggregation, inference) side by side.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Prerequisites](#prerequisites)
3. [Python Environment Setup](#python-environment-setup)
4. [Java Build](#java-build)
5. [Running Locally — All 3 Nodes on One Machine](#running-locally--all-3-nodes-on-one-machine)
6. [Running on Multiple Machines (LAN)](#running-on-multiple-machines-lan)
7. [Solo Mode — Single Node Testing](#solo-mode--single-node-testing)
8. [Testing a Prediction](#testing-a-prediction)
9. [Port Reference](#port-reference)
10. [Troubleshooting](#troubleshooting)
11. [Dependency Version Lock](#dependency-version-lock)

---

## How It Works

```
Each Hospital Node
┌─────────────────────────────────────────────┐
│  Java Coordinator (port 6001/6002/6003)      │  ← P2P TCP with other Java nodes
│    - Round orchestration                     │
│    - Broadcasts ROUND_START / ROUND_COMPLETE │
│    - Calls local Python via HTTP             │
│                                              │
│  Python ML Server                            │
│    - IPC server  (port 5101/5102/5103)       │  ← Java talks here
│    - Weight server (port 5201/5202/5203)     │  ← Other Python nodes pull adapters here
│    - Trains LoRA adapter on local data       │
│    - Runs FedIT aggregation (SVD merge)      │
│    - Serves symptom predictions              │
└─────────────────────────────────────────────┘
```

**One round =** local training → round barrier → FedIT merge → merged adapter reloaded → next round.

No raw patient data ever leaves a node. Only small LoRA adapter files (~5MB) travel across the network.

---

## Prerequisites

### System Requirements
- **Java 17+** (OpenJDK or Oracle JDK)
- **Maven 3.8+**
- **Python 3.10 or 3.11** (3.12 not recommended — some PEFT versions have issues)
- **Anaconda or Miniconda** (strongly recommended — see why below)
- ~4GB RAM per node for CPU training
- ~2GB disk for model cache + adapter checkpoints

### Check your versions
```bash
java -version      # need 17+
mvn -version       # need 3.8+
python --version   # need 3.10 or 3.11
conda --version    # need 23+
```

---

## Python Environment Setup

> **Why conda?**
> PyTorch, Transformers, and PEFT have strict interdependencies.
> Installing into your system Python or a generic venv frequently causes
> silent version conflicts that produce cryptic runtime errors.
> A named conda environment pins everything in one place and is
> fully reproducible across machines.

### Step 1 — Create a locked conda environment

```bash
conda create -n dfl-project python=3.11 -y
conda activate dfl-project
```

### Step 2 — Install PyTorch (CPU version, stable)

```bash
# CPU — works on any machine including WSL2
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

> **Intel Arc GPU (Windows native only)**
> ```bash
> pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu \
>     intel-cmplr-lib-rt intel-cmplr-lib-ur intel-sycl-rt pytorch-triton-xpu \
>     --index-url https://download.pytorch.org/whl/xpu \
>     --extra-index-url https://pypi.org/simple
> python -c "import torch; print(torch.xpu.is_available())"  # should print True
> ```

### Step 2.5 — Fix shared library path (WSL2 / Linux only)

After activating the conda environment, run this before starting any Python server:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

To make this permanent so you don't have to run it every session:

```bash
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

> **Why this is needed:** PyTorch and PEFT install native `.so` shared libraries
> inside the conda environment's `lib/` folder. Linux's dynamic linker does not
> search conda paths by default, causing `ImportError` or `OSError` on startup
> if this variable is not set.

> libgomp.so.1: cannot open shared object file or GLIBCXX not found | Run export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH — linker can't find conda native libs 

### Step 3 — Install ML libraries with pinned versions

```bash
pip install \
    transformers==4.44.2 \
    peft==0.12.0 \
    datasets==2.20.0 \
    accelerate==0.33.0 \
    safetensors==0.4.4 \
    flask==3.0.3 \
    requests==2.32.3
```

> **Why pin versions?**
> `transformers`, `peft`, and `accelerate` release frequently and break each other's APIs.
> The versions above are tested together. Using `pip install transformers` (unpinned)
> will likely give you a version that is incompatible with the `peft` version
> that also gets pulled in. Always pin in a shared project.

### Step 4 — Save the environment for teammates

```bash
# After installing — export exact versions for reproducibility
pip freeze > requirements.txt
```

Teammates can then reproduce the exact environment:
```bash
conda create -n dfl-project python=3.11 -y
conda activate dfl-project
pip install -r requirements.txt
```

### Step 5 — Verify the environment

```bash
python -c "
import torch, transformers, peft, datasets, flask, safetensors
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('datasets:', datasets.__version__)
print('flask:', flask.__version__)
print('safetensors:', safetensors.__version__)
print('All OK')
"
```

---

## Java Build

```bash
# From project root (where pom.xml is)
mvn clean package -q

# Verify the JAR was built and has a Main-Class
unzip -p target/hospitaldfl-1.0-SNAPSHOT.jar META-INF/MANIFEST.MF
# You should see: Main-Class: hospital.dfl.MainNode
```

The build produces a **fat JAR** — Gson and all dependencies are bundled inside.
You only need to ship this one `.jar` file to other machines.

---

## Running Locally — All 3 Nodes on One Machine

You will need **6 terminal windows** (or tmux panes): 3 for Python, 3 for Java.

### Step 1 — Activate conda environment in every terminal

```bash
conda activate dfl-project
```

### Step 2 — Start all 3 Python servers (do this FIRST)

```bash
# Terminal 1
python python/fl_server.py 1

# Terminal 2
python python/fl_server.py 2

# Terminal 3
python python/fl_server.py 3
```

Each server prints two lines on startup:
```
[Node 1] IPC server   → 127.0.0.1:5101
[Node 1] Weight server → 0.0.0.0:5201
```

### Step 3 — Verify Python servers are up

```bash
curl http://127.0.0.1:5101/status   # {"phase":"idle","round":0}
curl http://127.0.0.1:5102/status
curl http://127.0.0.1:5103/status
```

All three must return `idle` before starting Java.

### Step 4 — Start all 3 Java coordinators

```bash
# Terminal 4
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 1 6001 6002 6003

# Terminal 5
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 2 6002 6001 6003

# Terminal 6
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 3 6003 6001 6002
```

**Argument format:** `<nodeId> <ownPort> <peerPort1> <peerPort2>`

You should see within a few seconds:
```
Started Java Coordinator Node 1 on port 6001
Waiting 5 seconds for peers to start...
=== Round 1 / 5 ===
[Java] Triggering local Python training...
```

---

## Running on Multiple Machines (LAN)

Each physical machine runs **one node** — one Python server + one Java coordinator.

### Step 1 — Find each machine's LAN IP

```bash
# Linux / Mac
ip addr show | grep "inet " | grep -v 127
# or
hostname -I

# Windows
ipconfig | findstr "IPv4"
```

Example: Machine 1 = `192.168.1.101`, Machine 2 = `192.168.1.102`, Machine 3 = `192.168.1.103`

### Step 2 — Update IP addresses in RoundController.java

```java
// In RoundController.java, update ALL_WEIGHT_ADDRS:
static {
    ALL_WEIGHT_ADDRS = new HashMap<>();
    ALL_WEIGHT_ADDRS.put(1, "192.168.1.101:5201");  // Machine 1
    ALL_WEIGHT_ADDRS.put(2, "192.168.1.102:5202");  // Machine 2
    ALL_WEIGHT_ADDRS.put(3, "192.168.1.103:5203");  // Machine 3
}
```

Then rebuild:
```bash
mvn clean package -q
```

### Step 3 — Open firewall ports on each machine

Each machine needs these ports open for inbound connections:

| Port | Purpose |
|---|---|
| 6001 / 6002 / 6003 | Java P2P TCP (Coordination Plane) |
| 5201 / 5202 / 5203 | Python weight file server (Weight Transfer Plane) |

```bash
# Linux (ufw)
sudo ufw allow 6001/tcp
sudo ufw allow 5201/tcp

# Linux (firewalld)
sudo firewall-cmd --add-port=6001/tcp --permanent
sudo firewall-cmd --add-port=5201/tcp --permanent
sudo firewall-cmd --reload

# Windows Defender Firewall
netsh advfirewall firewall add rule name="DFL Java" dir=in action=allow protocol=TCP localport=6001
netsh advfirewall firewall add rule name="DFL Python" dir=in action=allow protocol=TCP localport=5201
```

### Step 4 — Start Python on each machine

```bash
# Machine 1
conda activate dfl-project
python python/fl_server.py 1

# Machine 2
conda activate dfl-project
python python/fl_server.py 2

# Machine 3
conda activate dfl-project
python python/fl_server.py 3
```

### Step 5 — Verify cross-machine connectivity

From Machine 1, test that it can reach Machine 2 and 3's weight servers:
```bash
curl http://192.168.1.102:5202/status  # should return 404 or weights not available
curl http://192.168.1.103:5203/status
```

### Step 6 — Start Java on each machine

```bash
# Machine 1
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 1 6001 6002 6003

# Machine 2
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 2 6002 6001 6003

# Machine 3
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 3 6003 6001 6002
```

> **Note:** For LAN mode, peer ports in the Java args refer to ports on **other machines**.
> Java connects to `127.0.0.1:<peerPort>` by default — you will need to update
> `ClientNode.java` to use the real peer IPs instead of `127.0.0.1` for a true LAN deployment.

---

## Solo Mode — Single Node Testing

Test the full training pipeline on one node with no peers:

```bash
# Terminal 1 — Python
conda activate dfl-project
python python/fl_server.py 1

# Terminal 2 — Java (no peer ports = solo mode)
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 1 6001
```

Solo mode skips P2P broadcasting and aggregation. Training completes locally and the adapter is saved. Useful for verifying the training pipeline before running a full cluster.

---

## Testing a Prediction

After at least one round completes, test the diagnosis endpoint:

```bash
curl -X POST http://127.0.0.1:5101/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "sudden weakness in left arm and leg, swollen right lower leg after long flight"}'
```

Expected response:
```json
{
  "node": 1,
  "symptoms": "sudden weakness in left arm...",
  "diagnosis": "The presence of a patent foramen ovale (PFO)..."
}
```

---

## Port Reference

| Node | Java TCP (Coordination) | Python IPC (Java→Python) | Python Weight Server (Peer→Peer) |
|---|---|---|---|
| Node 1 | 6001 | 5101 | 5201 |
| Node 2 | 6002 | 5102 | 5202 |
| Node 3 | 6003 | 5103 | 5203 |

**Formula:**
- Java TCP port = `6000 + nodeId`
- Python IPC port = `5100 + nodeId`
- Python Weight port = `5200 + nodeId`

---

## Troubleshooting

### `no main manifest attribute`
The shade plugin is in `<pluginManagement>` instead of `<build><plugins>`.
Move the entire `maven-shade-plugin` block into `<build><plugins>` directly.

### `Timeout waiting for Python status: training_complete`
1. Check Python server is actually running: `curl http://127.0.0.1:5101/status`
2. If `Connection refused` → Python server crashed or was not started
3. If returns `{"phase":"training"}` for a long time → training is still running (check Python terminal for progress bar)
4. If returns `{"phase":"error"}` → check Python terminal for the actual exception

### `Connection refused` between Java nodes
- Other Java nodes are not started yet — start them within 5 seconds of each other
- Wrong peer port in args — check argument order: `<nodeId> <ownPort> <peer1> <peer2>`
- For LAN: firewall blocking port — open ports 6001-6003

### `fan_in_fan_out is set to False` warning
Normal for GPT-2 + LoRA. PEFT auto-corrects this. Safe to ignore.

### `pin_memory... no accelerator found` warning
You are training on CPU. `pin_memory=True` only benefits GPU training.
Set `dataloader_pin_memory=False` in `TrainingArguments` to silence it.

### Training is extremely slow (>10s per step)
- On CPU this is expected. Use `max_steps=30` for pipeline testing.
- On WSL2: reduce `per_device_train_batch_size=1` and `dataloader_num_workers=0`
- On Windows native with Intel Arc: install `torch==2.7.1+xpu` for GPU acceleration

### `Non contiguous tensor` error in aggregation
All tensor outputs from SVD operations must have `.contiguous()` called before `save_file()`.
This is already fixed in `aggregator.py` — if you see this, check that you have the latest version.

### `does not require grad and does not have a grad_fn` in Round 2+
`PeftModel.from_pretrained()` was called without `is_trainable=True`.
The loaded adapter is in inference mode and cannot be trained.
Fix: `PeftModel.from_pretrained(base, OUTPUT_DIR, is_trainable=True)`

### `OutOfMemoryError` (Python process killed silently on WSL2)
WSL2 has limited RAM. Run with reduced settings:
```python
per_device_train_batch_size=1
gradient_accumulation_steps=4
dataloader_num_workers=0
dataloader_pin_memory=False
max_steps=30
```
Or switch to native Windows which has access to full system RAM.

---

## Dependency Version Lock

These exact versions are tested and known to work together:

```
# requirements.txt
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
transformers==4.44.2
peft==0.12.0
datasets==2.20.0
accelerate==0.33.0
safetensors==0.4.4
flask==3.0.3
requests==2.32.3
```

**Common version conflict warnings:**

| Symptom | Likely cause |
|---|---|
| `ImportError: cannot import name 'LoraConfig'` | `peft` version too old — needs 0.6+ |
| `TypeError` in `TrainingArguments` | `transformers` and `accelerate` version mismatch — update both together |
| `safetensors.torch has no attribute load_file` | `safetensors` version too old — needs 0.3+ |
| `torch.linalg.svd` missing | `torch` version too old — needs 1.9+ |
| `PeftModel.from_pretrained` missing `is_trainable` arg | `peft` version too old — needs 0.6+ |

**Updating safely:**
Never run `pip install transformers --upgrade` alone.
Always update the interdependent group together:
```bash
pip install transformers==X.X.X peft==X.X.X accelerate==X.X.X --upgrade
```
Then test before committing `requirements.txt`.
