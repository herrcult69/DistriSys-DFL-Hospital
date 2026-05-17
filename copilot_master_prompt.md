# Distributed Federated Learning System — Master Context Prompt
## For: GitHub Copilot / AI Assistant Continuation Sessions

---

## 1. PROJECT OVERVIEW

**Project:** Distributed Federated Learning (DFL) Hospital Symptom Checker  
**Status:** Working prototype with end-to-end 3-node round orchestration, local LoRA training, FedIT aggregation, atomic merged-adapter overwrite, and round-to-round adapter reuse.  
**Goal:** 3 simulated hospital nodes each train a local GPT-2 LoRA model on private medical QA/CoT data, then collaborate to create a globally improved adapter without sharing raw patient data.

**Core architectural rule:**  
- **Java owns coordination and networking.**
- **Python owns all ML operations.**
- **Java never manipulates tensors or model weights.**

---

## 2. CANONICAL TERMS

Use these exact terms when discussing the system:

| Term | Definition |
|---|---|
| **Coordinator Node** | The Java process on each machine. Owns P2P TCP messaging, round orchestration, and user-facing query routing. Never touches tensors. |
| **Local Adapter Server** | The Python Flask service (`fl_server.py`) on each node. Owns training, aggregation trigger handling, inference, and weight serving. |
| **Coordination Plane** | Java↔Java raw TCP channel for control messages only, such as `ROUND_START` and `ROUND_COMPLETE`. |
| **IPC Channel** | Java→Python HTTP channel on localhost. Used for `/train`, `/status`, `/aggregate`, and `/predict`. |
| **Weight Transfer Plane** | Python↔Python HTTP channel used for downloading `adapter_model.safetensors` from peer nodes. |
| **Local Adapter** | The node’s own LoRA adapter checkpoint stored as `output/p{nodeId}_gpt2_lora/adapter_model.safetensors`. |
| **Global Adapter** | The merged adapter produced by FedIT aggregation and atomically written back into the node’s local adapter path before the next round. |
| **Adapter Checkpoint** | The `adapter_model.safetensors` + `adapter_config.json` pair saved by PEFT. |
| **FedIT Merge** | The correct LoRA aggregation strategy: average \(B_i @ A_i\) per layer, then SVD re-decompose into merged `lora_A` and `lora_B`. |
| **SVD Re-decomposition** | The step `U, S, Vh = svd(delta_avg)` followed by `B_merged = U[:, :r] * sqrt(S)` and `A_merged = diag(sqrt(S)) @ Vh[:r, :]`. |
| **Round** | One FL cycle: local training → TCP barrier → aggregation → merged adapter overwrite → next round starts from merged adapter. |
| **Aggregator Node** | The node whose Java coordinator calls local Python `/aggregate` for the current round. In practice this can be any node, but the current prototype commonly uses Node 1. |
| **Model Cache** | The in-memory GPT-2/PEFT model cached by `fl_server.py` for `/predict`. Invalidated after aggregation so future inference reloads merged weights. |
| **Length-Prefix Framing** | TCP protocol format: 4-byte big-endian length followed by UTF-8 JSON bytes. Prevents stream-splitting bugs. |
| **Solo Mode** | Single-node mode where no peers exist, so training can run locally without aggregation. |

---

## 3. PORT MAP

### Single-machine prototype
| Node | Java TCP | Java→Python IPC | Python Weight Server |
|---|---:|---:|---:|
| Node 1 | 6001 | 5101 | 5201 |
| Node 2 | 6002 | 5102 | 5202 |
| Node 3 | 6003 | 5103 | 5203 |

### Port formulas
- **IPC port** = `5100 + nodeId`
- **Weight port** = `5200 + nodeId`

For real LAN deployment:
- Replace `127.0.0.1` in `ALL_WEIGHT_ADDRS` with each machine’s LAN IP.
- Keep the same port formulas unless firewall or IT policy requires different ports.

---

## 4. TECHNOLOGY STACK

### Python / ML
| Component | Technology | Notes |
|---|---|---|
| Base model | `distilbert/distilgpt2` | Smaller than full GPT-2, more practical for CPU prototype |
| Fine-tuning | LoRA via PEFT | `r=8`, `lora_alpha=32`, `target_modules=["c_attn"]`, `lora_dropout=0.1`, `bias="none"` |
| ML framework | PyTorch + Transformers | Current prototype runs on CPU |
| Aggregation | FedIT-style product averaging + SVD | Implemented in `aggregator.py` |
| Dataset | JSONL | Fields: `Question`, `Complex_CoT`, `Response` |
| Inference / serving | Flask | Two Flask apps inside one process: local IPC app + weight transfer app |
| Weight format | `safetensors` | All tensors must be contiguous before save |

### Java / coordination
| Component | Technology | Notes |
|---|---|---|
| P2P networking | Raw TCP sockets | JSON payloads with 4-byte length prefix |
| HTTP client | `HttpURLConnection` | Used for Python IPC |
| Concurrency | Threads + `LinkedBlockingQueue` | `ConnectionHandler` feeds `MessageStore` |
| Build | Maven + Shade plugin | Fat JAR with Main-Class written into manifest |
| JSON library | Gson 2.11.0 | Present as dependency, not yet fully used in parsing |

---

## 5. FILE STRUCTURE

```text
project-root/
├── src/main/java/hospital/dfl/
│   ├── MainNode.java
│   ├── RoundController.java
│   ├── ServerNode.java
│   ├── ConnectionHandler.java
│   ├── ClientNode.java
│   ├── MessageStore.java
│   └── NodeParam.java
├── python/
│   ├── fl_server.py
│   ├── local_trainer.py
│   └── aggregator.py
├── dataset_part_1.jsonl
├── dataset_part_2.jsonl
├── dataset_part_3.jsonl
├── output/
│   ├── p1_gpt2_lora/
│   ├── p2_gpt2_lora/
│   ├── p3_gpt2_lora/
│   └── merged_adapter_round_<round>_<pid>.safetensors
└── pom.xml
```

---

## 6. COMPLETE ROUND LIFECYCLE

```text
Round N
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Java RoundController broadcasts TCP:
   {"type":"ROUND_START","round":N}

2. Java calls local Python:
   POST http://127.0.0.1:(5100+nodeId)/train
   body: {"round": N}

3. Python local_app starts background training thread:
   - Round 1: initialize fresh LoRA adapter
   - Round > 1: load existing merged adapter from OUTPUT_DIR
                using PeftModel.from_pretrained(..., is_trainable=True)
                then re-enable LoRA grads as safety net

4. Java polls:
   GET /status
   until phase == "training_complete"

5. Java broadcasts TCP:
   {"type":"ROUND_COMPLETE","node":<nodeId>,"round":N}

6. Java waits for peer ROUND_COMPLETE messages:
   - self is pre-added to the finished set
   - no self-send is needed

7. Java triggers local aggregation:
   POST /aggregate
   body includes:
   - round number
   - peer weight server map excluding self

8. Python aggregation thread:
   - loads own adapter from disk
   - downloads peer adapters from /weights
   - performs FedIT merge: avg(B_i @ A_i) + SVD
   - calls .contiguous() on tensors
   - saves archive copy:
     output/merged_adapter_round_<round>_<pid>.safetensors
   - atomically overwrites:
     output/p{nodeId}_gpt2_lora/adapter_model.safetensors
   - invalidates inference model cache

9. Java polls until phase == "aggregate_complete"

10. Next round starts using merged adapter as the training base
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 7. CURRENT IMPLEMENTATION DETAILS

### `aggregator.py`
- Own adapter is loaded from disk directly, not downloaded via HTTP.
- Peer adapters are downloaded from `/weights`.
- FedIT uses product averaging \(B_i @ A_i\), not naive averaging of `lora_A` and `lora_B` separately.
- SVD outputs are forced to `.contiguous()` before saving.
- Save path behavior:
  1. archive copy → `output/merged_adapter_round_<round>_<pid>.safetensors`
  2. atomic overwrite → `output/p{nodeId}_gpt2_lora/adapter_model.safetensors`

### `fl_server.py`
- Runs two Flask apps:
  - `local_app` on `127.0.0.1:<5100+nodeId>`
  - `weight_app` on `0.0.0.0:<5200+nodeId>`
- Uses `_status_lock` to protect the shared status dict.
- Uses `_model_lock` to protect the inference model cache.
- `/predict` uses cached model loading and reloads after cache invalidation.
- `/aggregate` invalidates the model cache after a successful merge.

### `local_trainer.py`
- Round 1 creates a fresh PEFT LoRA model.
- Round > 1 loads the merged adapter from `OUTPUT_DIR`.
- `PeftModel.from_pretrained(..., is_trainable=True)` is required.
- `_enable_lora_grads()` is used as a safety net so LoRA params definitely require gradients.
- Dataset text format is:
  - `Question: ...`
  - `Reasoning: ...`
  - `Answer: ...`

### Java coordinator
- TCP uses length-prefix framing.
- `buildPeersJson()` excludes self from weight download peers.
- `waitForAllPeers()` pre-adds self to the finished set.
- `pollUntilStatus()` currently waits for target `"phase"` text in `/status` response.
- `RoundController` timeouts are:
  - training: 60 minutes
  - aggregation: 15 minutes

---

## 8. KNOWN FIXES ALREADY APPLIED

| Problem | Fix now in code |
|---|---|
| `no main manifest attribute` | Shade plugin moved into active `<build><plugins>` config |
| Port collision on Python server | IPC = `5100 + nodeId`, weight server = `5200 + nodeId` |
| Self-download bug during aggregation | Self adapter loaded from disk instead of HTTP |
| Wrong peer JSON for aggregation | Java excludes self from `peers` payload |
| TCP message fragmentation risk | 4-byte length-prefix framing implemented |
| Waiting barrier self-message bug | Self pre-added to finished set in `waitForAllPeers()` |
| `can only concatenate list not str` | Safer tokenization and non-batched map path |
| `too many indices for tensor of dimension 2` | Example-by-example tokenization path |
| Non-contiguous tensor save failure | `.contiguous()` applied before `save_file()` |
| Round 2 gradient failure | `is_trainable=True` + `_enable_lora_grads()` |
| Partial file read race | Aggregated adapter written atomically via `.tmp` then `os.replace()` |
| Stale inference model after merge | `_invalidate_model_cache()` after successful aggregation |

---

## 9. CURRENT TRAINING CONFIGURATION

### What the code is using right now
```python
TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    max_steps=3,
    save_strategy="no",
    learning_rate=2e-4,
    logging_steps=10,
    report_to="none",
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
)
```

### What this means
- `max_steps=3` is **testing mode only**
- `num_train_epochs=2` is effectively irrelevant once `max_steps` is reached
- current config is for **pipeline verification**, not real learning quality

### Recommended configs
**Quick pipeline demo**
```python
max_steps=3 to 10
per_device_train_batch_size=4
```

**Safer WSL / low-memory demo**
```python
per_device_train_batch_size=1
gradient_accumulation_steps=4
max_steps=30 to 100
num_train_epochs=1
```

**Real prototype demo**
```python
per_device_train_batch_size=1
gradient_accumulation_steps=4
max_steps=300
num_train_epochs=1
TOTAL_ROUNDS=5
```

---

## 10. STARTUP SEQUENCE

```bash
# 1. Start Python servers first
python fl_server.py 1
python fl_server.py 2
python fl_server.py 3

# 2. Verify
curl http://127.0.0.1:5101/status
curl http://127.0.0.1:5102/status
curl http://127.0.0.1:5103/status

# 3. Build Java
mvn clean package -q

# 4. Start Java nodes
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 1 6001 6002 6003
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 2 6002 6001 6003
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 3 6003 6001 6002
```

### Solo mode
```bash
python fl_server.py 1
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 1 6001
```

---

## 11. WHAT WORKS NOW

### Confirmed working
- 3-node Java coordination
- Python local training trigger
- TCP round signaling
- Round barrier synchronization
- Peer adapter collection
- FedIT aggregation with SVD
- Atomic merged adapter save
- Round-to-round adapter reuse
- `/weights` endpoint
- `/status` endpoint
- `/predict` endpoint with cache invalidation

---

## 12. KNOWN REMAINING ISSUES

| Area | Current issue |
|---|---|
| `pollUntilStatus()` | It does not fast-fail on `"training_error"` or `"aggregate_error"`; it can still timeout instead of surfacing Python error immediately |
| JSON parsing in Java | Still uses `String.contains()` and substring parsing; should move to Gson |
| Peer liveness | No heartbeat or explicit health check between Java nodes |
| `/predict` prompt | Inference still uses `Symptoms: ... Diagnosis:` style, while training used `Question/Reasoning/Answer`; prompt alignment should be improved |
| `/predict` decode | Current inference helper decodes the entire generated sequence, so it may echo the prompt |
| Resource profile | Current `batch_size=4` may still be too heavy on WSL2 for longer runs |
| Native Windows / Intel Arc | Not fully enabled yet; still mostly CPU prototype |

---

## 13. NEXT TASKS FOR THE NEXT SESSION

1. Update `/predict` to use the same prompt family as training:
   - `Question: ...`
   - `Reasoning:`
   - then generate the answer
2. Slice off prompt tokens when decoding generation so `/predict` returns only the answer text.
3. Make `pollUntilStatus()` fail fast on:
   - `training_error`
   - `aggregate_error`
4. Replace Java string parsing with Gson objects.
5. Add heartbeat or readiness checks for peers.
6. Decide whether aggregation should happen on every node or only on a designated aggregator node.
7. Increase training realism after pipeline validation by moving from `max_steps=3` to a larger run.

---

## 14. IMPORTANT CONSTRAINTS

- Java never manipulates model tensors.
- Python owns training, aggregation, inference, and weight file I/O.
- Raw patient data never leaves a node.
- Only adapter weights move across the Weight Transfer Plane.
- Always call `.contiguous()` before saving merged tensors with `safetensors`.
- For continued training, `PeftModel.from_pretrained()` must use `is_trainable=True`.
- Python servers must start before Java coordinators.
- The shade plugin must stay in `<build><plugins>`, not only in `<pluginManagement>`.

---

## 15. IF ASKED TO MODIFY THE SYSTEM

When making changes, preserve these invariants:
1. Keep Java/Python separation of concerns.
2. Do not reintroduce Java-side weight math.
3. Do not replace FedIT with naive `avg(A)` and `avg(B)`.
4. Do not remove atomic adapter overwrite.
5. Do not remove length-prefix framing from TCP.
6. Do not let a node HTTP-download its own adapter.