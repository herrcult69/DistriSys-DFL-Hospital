# Distributed Federated Learning System — Master Context Prompt
## For: GitHub Copilot / AI Assistant Continuation Sessions

***

## 1. PROJECT OVERVIEW

**Project:** Distributed Federated Learning (DFL) Hospital Symptom Checker  
**Status:** ✅ Working Prototype — All 3 nodes complete full FL rounds end-to-end  
**Goal:** 3 simulated hospital nodes each train a local GPT-2 LoRA model on private patient data, then collaborate via FedIT aggregation to produce a globally smarter diagnostic model — without any node ever sharing raw patient data.

***

## 2. SYSTEM TOPOLOGY (CANONICAL TERMS)

Use these exact terms when discussing the system:

| Term | Definition |
|---|---|
| **Coordinator Node** | The Java process on each machine. Owns P2P TCP communication, round orchestration, and user query routing. Never touches tensors. |
| **Local Adapter Server** | The Flask Python process (`fl_server.py`) running on each machine. Owns all ML work. Two sub-servers per node. |
| **Coordination Plane** | The Java↔Java TCP channel. Carries JSON control messages only (ROUND_START, ROUND_COMPLETE). Port range: 6001–6003. |
| **IPC Channel** | The Java→Python HTTP channel on localhost only. Triggers training and prediction. Port formula: `5100 + nodeId` (5101, 5102, 5103). |
| **Weight Transfer Plane** | The Python↔Python HTTP channel for binary adapter file transfer. Port formula: `5200 + nodeId` (5201, 5202, 5203). Binds `0.0.0.0` for LAN access. |
| **Local Adapter** | The `adapter_model.safetensors` file in each node's output dir after local training. e.g. `output/p1_gpt2_lora/adapter_model.safetensors` |
| **Global Adapter** | The merged `adapter_model.safetensors` produced by FedIT aggregation. Overwrites each node's Local Adapter atomically before Round N+1. |
| **Adapter Checkpoint** | The `adapter_config.json` + `adapter_model.safetensors` pair saved after each training round. |
| **FedIT Merge** | The correct FedAvg algorithm for LoRA: compute `avg(B_i @ A_i)` per layer, then SVD re-decompose to get new A/B matrices. NOT naive avg(A) / avg(B). |
| **SVD Re-decomposition** | The mathematical step inside FedIT: `U, S, Vh = svd(delta_avg)`, then `B_merged = U[:,:r] * sqrt(S)`, `A_merged = diag(sqrt(S)) @ Vh[:r,:]`. |
| **Round** | One complete FL cycle: local training → ROUND_COMPLETE broadcast → FedIT merge → Global Adapter reload. |
| **Aggregator Node** | The node that runs `collect_and_aggregate()`. In this prototype: whichever node Java tells to `/aggregate` — currently Node 1. |
| **Model Cache** | The in-memory loaded PeftModel in `fl_server.py`. Invalidated after each FedIT merge so `/predict` uses the latest Global Adapter. |
| **Length-Prefix Framing** | The TCP message protocol: 4-byte big-endian integer length header followed by UTF-8 JSON body. Prevents stream fragmentation. |
| **Solo Mode** | Single-node operation: no P2P broadcast, no aggregation, local training only. Activated when `peerPorts` list is empty. |

***

## 3. PORT MAP (SINGLE MACHINE PROTOTYPE)

| Node | Coordinator (Java TCP) | IPC Channel (Java→Python) | Weight Transfer (Python→Python) |
|---|---|---|---|
| Node 1 | 6001 | 5101 | 5201 |
| Node 2 | 6002 | 5102 | 5202 |
| Node 3 | 6003 | 5103 | 5203 |

For LAN deployment: replace `127.0.0.1` with each machine's LAN IP in `ALL_WEIGHT_ADDRS` (RoundController.java) and peer port args.

***

## 4. TECHNOLOGY STACK

### Python / ML Stack
| Component | Technology | Notes |
|---|---|---|
| Base Model | `distilbert/distilgpt2` | ~82M params, ~330MB. Smaller than full GPT-2 for faster CPU training |
| Fine-tuning | LoRA via HuggingFace PEFT | `r=8, lora_alpha=32, target_modules=["c_attn"], bias="none"` |
| Framework | PyTorch + HuggingFace Transformers | CPU training; XPU (Intel Arc) supported with `torch==2.7.1+xpu` |
| Aggregation | FedIT (SVD re-decomposition) | In `aggregator.py`. NOT naive FedAvg on raw A/B matrices |
| Data Format | JSONL | Fields: `Question`, `Complex_CoT`, `Response` (medical diagnostic CoT dataset) |
| Web Server | Flask (dual-server architecture) | `local_app` on 127.0.0.1 (IPC), `weight_app` on 0.0.0.0 (weight transfer) |
| Weight Format | `safetensors` | Binary safe format. All tensors must be `.contiguous()` before `save_file()` |

### Java / Coordination Stack
| Component | Technology | Notes |
|---|---|---|
| P2P Protocol | Raw TCP with 4-byte length-prefix framing | `DataOutputStream.writeInt(len)` + `DataInputStream.readFully()` |
| Message Format | JSON strings over TCP | Parsed with `String.contains()` for prototype; Gson available for upgrade |
| Server | Multithreaded `ServerSocket` | `ServerNode` accepts, `ConnectionHandler` handles each connection in new thread |
| Message Queue | `LinkedBlockingQueue` in `MessageStore` | Thread-safe message passing between `ConnectionHandler` and `RoundController` |
| HTTP Client | `HttpURLConnection` | Used for IPC Channel and status polling |
| Build | Maven with `maven-shade-plugin` | Fat JAR. `Main-Class` in MANIFEST.MF via `ManifestResourceTransformer` |
| Dependencies | Gson 2.11.0 | Available but not yet used — raw string parsing used for prototype |

***

## 5. FILE STRUCTURE

```
project-root/
├── src/main/java/hospital/dfl/
│   ├── MainNode.java           — Entry point, starts ServerNode + RoundController
│   ├── RoundController.java    — FL round orchestration, HTTP calls to Python, TCP broadcast
│   ├── ServerNode.java         — TCP ServerSocket listener
│   ├── ConnectionHandler.java  — Per-connection thread, writes to MessageStore
│   ├── ClientNode.java         — TCP client, sends length-prefixed JSON to peers
│   ├── MessageStore.java       — LinkedBlockingQueue for inter-thread message passing
│   └── NodeParam.java          — Immutable node config (nodeId, port, peerPorts)
├── python/
│   ├── fl_server.py            — Dual Flask server (IPC + Weight Transfer)
│   ├── local_trainer.py        — GPT-2 LoRA training logic
│   └── aggregator.py           — FedIT merge (SVD re-decomposition)
├── dataset_part_1.jsonl        — Node 1's local hospital data split
├── dataset_part_2.jsonl        — Node 2's local hospital data split
├── dataset_part_3.jsonl        — Node 3's local hospital data split
├── output/
│   ├── p1_gpt2_lora/           — Node 1 adapter checkpoint (Local Adapter)
│   ├── p2_gpt2_lora/           — Node 2 adapter checkpoint
│   ├── p3_gpt2_lora/           — Node 3 adapter checkpoint
│   └── merged_adapter_round_N_PID.safetensors  — Archive of each Global Adapter
└── pom.xml                     — Maven build. Shade plugin in <plugins> NOT <pluginManagement>
```

***

## 6. COMPLETE ROUND LIFECYCLE

```
Round N
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Java] RoundController.runRound(N)
  │
  ├─ broadcastTCP(ROUND_START) ──TCP──► peer Java nodes (6002, 6003)
  │
  ├─ httpPost(5101/train, {round:N}) ──HTTP──► fl_server.py local_app
  │     └─ Thread: local_trainer.train(N)
  │           ├─ Round 1: fresh LoRA init via get_peft_model()
  │           └─ Round 2+: PeftModel.from_pretrained(is_trainable=True)
  │                        + _enable_lora_grads() safety net
  │
  ├─ pollUntilStatus("training_complete") every 2s, timeout 60min
  │
  ├─ broadcastTCP(ROUND_COMPLETE, node:1) ──TCP──► peers
  │
  ├─ waitForAllPeers("ROUND_COMPLETE") — blocks on MessageStore queue
  │     └─ Self pre-added to finishedNodes (no self-send needed)
  │
  ├─ httpPost(5101/aggregate, {round:N, peers:{2:"127.0.0.1:5202",...}})
  │     └─ Thread: aggregator.collect_and_aggregate()
  │           ├─ Load own adapter from disk (no HTTP to self)
  │           ├─ HTTP GET 5202/weights ──► Node 2 weight_app
  │           ├─ HTTP GET 5203/weights ──► Node 3 weight_app
  │           ├─ _fedit_merge(): avg(B_i@A_i) per layer → SVD → new A,B
  │           ├─ .contiguous() on all tensors before save_file()
  │           ├─ save archive: output/merged_adapter_round_N_PID.safetensors
  │           └─ atomic write: .tmp → os.replace() → adapter_model.safetensors
  │
  ├─ pollUntilStatus("aggregate_complete") timeout 15min
  │
  └─ _invalidate_model_cache() → next /predict reloads Global Adapter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Round N+1 starts — nodes now training FROM Global Adapter
```

***

## 7. KNOWN FIXES ALREADY APPLIED

All of these bugs have been fixed in the current codebase:

| Bug | Fix Applied |
|---|---|
| `no main manifest attribute` | `maven-shade-plugin` moved from `<pluginManagement>` to `<plugins>` |
| `can only concatenate list not str` | `batched=False` in `dataset.map()` + `list(result["input_ids"])` for labels |
| `too many indices for tensor of dimension 2` | `batched=False` eliminates nested list → 3D tensor issue |
| Non-contiguous tensor in `save_file()` | `.contiguous()` on all SVD outputs + blanket pass before `save_file()` |
| Round 2 `does not require grad` error | `PeftModel.from_pretrained(is_trainable=True)` + `_enable_lora_grads()` |
| All nodes timeout on single-node test | Solo mode: empty `peerPorts` skips P2P and aggregation |
| Python server OOM kill in WSL2 | `batch_size=1`, `gradient_accumulation_steps=4`, `dataloader_num_workers=0`, `pin_memory=False` |
| Port collision (all nodes on 5100) | Port formula: IPC=`5100+nodeId`, Weight=`5200+nodeId` |

***

## 8. STARTUP SEQUENCE (CRITICAL ORDER)

```bash
# ── Step 1: Start all 3 Python servers FIRST ──
python fl_server.py 1   # IPC:5101, Weights:5201
python fl_server.py 2   # IPC:5102, Weights:5202
python fl_server.py 3   # IPC:5103, Weights:5203

# ── Step 2: Verify all are up ──
curl http://127.0.0.1:5101/status   # {"phase":"idle","round":0}
curl http://127.0.0.1:5102/status
curl http://127.0.0.1:5103/status

# ── Step 3: Build Java (once) ──
mvn clean package -q

# ── Step 4: Start all 3 Java nodes ──
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 1 6001 6002 6003
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 2 6002 6001 6003
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 3 6003 6001 6002

# ── Solo mode (single node testing) ──
java -jar target/hospitaldfl-1.0-SNAPSHOT.jar 1 6001
```

***

## 9. TRAINING CONFIGURATION

```python
# Current settings in local_trainer.py (testing mode)
TrainingArguments(
    max_steps=3,                    # TESTING ONLY — remove for real training
    per_device_train_batch_size=1,  # Low for CPU/WSL2 memory
    gradient_accumulation_steps=4,  # Simulates effective batch size 4
    num_train_epochs=1,             # 1 epoch per round reduces client drift
    learning_rate=2e-4,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
)

# For real FL demo (30 min per round on CPU):
#   max_steps=300, batch_size=1, grad_accum=4, epochs=1, rounds=5

# LoRA config (do not change — adapter files depend on this)
LoraConfig(r=8, lora_alpha=32, target_modules=["c_attn"], lora_dropout=0.1, bias="none")
```

***

## 10. WHAT WORKS / WHAT IS NEXT

### ✅ Currently Working
- Full 3-node FL round lifecycle (train → broadcast → aggregate → reload)
- FedIT correct aggregation with SVD re-decomposition
- Atomic adapter file writes (no race conditions)
- Model cache invalidation after each merge
- Solo mode for single-node testing
- `/predict` endpoint with model cache
- Length-prefix framing on all TCP messages

### 🔧 Known Remaining Issues
- `pollUntilStatus` doesn't distinguish `Connection refused` (Python down) from timeout — fails after full timeout instead of fast-failing
- Java's `waitForAllPeers` uses `String.contains()` for JSON parsing — brittle, should use Gson
- No heartbeat between Java nodes — dead peer goes undetected until round barrier hangs
- `/predict` inference prompt format (`Symptoms: X\nDiagnosis:`) vs training format (`Question: X\nReasoning: Y\nAnswer:`) — slight mismatch, update predict to use training format
- Intel Arc XPU not yet enabled — currently CPU only. Requires `torch==2.7.1+xpu` from `https://download.pytorch.org/whl/xpu`

### 🚀 Next Steps (Suggested Priority)
1. Fix `/predict` prompt to match training format
2. Add fast-fail on `Connection refused` in `pollUntilStatus`
3. Switch from `String.contains()` to Gson parsing in `waitForAllPeers`
4. Enable Intel Arc XPU on native Windows
5. Add a simple CLI query interface in Java so users can type symptoms
6. Increase `max_steps` to 300 for a real demo run

***

## 11. AGGREGATION ALGORITHM DETAIL

```python
# FedIT — mathematically correct LoRA aggregation
# DO NOT replace with naive avg(A) / avg(B) — that is INCORRECT

def _fedit_merge(state_dicts, rank=8):
    for each LoRA layer prefix:
        # Step 1: Compute mean of products (not product of means)
        delta_avg = mean(B_i @ A_i for each node i)
        
        # Step 2: SVD re-decomposition
        U, S, Vh = torch.linalg.svd(delta_avg, full_matrices=False)
        sqrt_S = sqrt(S[:rank].clamp(min=1e-8))
        
        # Step 3: Reconstruct A and B — MUST call .contiguous() before save
        B_merged = (U[:, :rank] * sqrt_S).contiguous().to(float16)
        A_merged = (diag(sqrt_S) @ Vh[:rank, :]).contiguous().to(float16)
    
    # All non-LoRA keys (biases): simple mean, also .contiguous()
```

***

## 12. IMPORTANT CONSTRAINTS

- **Java never touches tensors** — it only sends JSON strings and timing signals
- **Python owns all ML** — training, aggregation, inference, file I/O
- **No raw patient data leaves nodes** — only Local Adapters travel on Weight Transfer Plane
- **`<pluginManagement>` does NOT run plugins** — shade plugin MUST be in `<build><plugins>`
- **Always call `.contiguous()` before `save_file()`** — SVD outputs are non-contiguous views
- **`PeftModel.from_pretrained()` for training MUST use `is_trainable=True`** — otherwise LoRA weights are frozen and training crashes
- **Start Python servers BEFORE Java** — Java polls `/status` immediately on startup