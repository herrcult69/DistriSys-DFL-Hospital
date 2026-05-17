import threading, os, sys, torch
from flask import Flask, request, jsonify, send_file
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

# ── Node identity (overridden by CLI args or env var) ─────────────────────────
NODE_ID     = int(os.environ.get("NODE_ID", 1))

# FIX (Bug 2): Offset ports by NODE_ID to avoid collision on single machine.
# Java's RoundController already computes: pythonBase = "http://127.0.0.1:" + (5100 + nodeId)
# and PEER_WEIGHT_ADDRS has 5201/5202/5203 — these formulas match.
LOCAL_PORT  = 5100 + NODE_ID   # Java IPC:    Node1→5101, Node2→5102, Node3→5103
WEIGHT_PORT = 5200 + NODE_ID   # Peer weight: Node1→5201, Node2→5202, Node3→5203
OUTPUT_DIR  = f"output/p{NODE_ID}_gpt2_lora"

# Allow full override from CLI: python fl_server.py <NODE_ID> [LOCAL_PORT] [WEIGHT_PORT]
if len(sys.argv) > 1:
    NODE_ID     = int(sys.argv[1])
    LOCAL_PORT  = int(sys.argv[2]) if len(sys.argv) > 2 else 5100 + NODE_ID
    WEIGHT_PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 5200 + NODE_ID
    OUTPUT_DIR  = f"output/p{NODE_ID}_gpt2_lora"

# Lazy-import trainer/aggregator AFTER NODE_ID is settled so their module-level
# constants (NODE_ID, OUTPUT_DIR, DATASET_PATH) are patched correctly.
import importlib, python.local_trainer as _lt, python.aggregator as _agg
_lt.NODE_ID      = NODE_ID
_lt.OUTPUT_DIR   = OUTPUT_DIR
_lt.DATASET_PATH = f"dataset_part_{NODE_ID}.jsonl"

# ── Shared status (FIX Issue 4): protected by a lock ─────────────────────────
_status_lock = threading.Lock()
_status = {"phase": "idle", "round": 0}

def _set_status(phase: str, round_num: int):
    with _status_lock:
        _status["phase"]  = phase
        _status["round"]  = round_num

def _get_status():
    with _status_lock:
        return dict(_status)

# ── FIX (Bug 4): Model cache — load once, reload only after aggregation ───────
_model     = None
_tokenizer = None
_model_lock = threading.Lock()

def _ensure_model_loaded():
    global _model, _tokenizer
    with _model_lock:
        if _model is None:
            _tokenizer = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            base = GPT2LMHeadModel.from_pretrained("distilbert/distilgpt2")
            adapter_path = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
            _model = PeftModel.from_pretrained(base, OUTPUT_DIR) if os.path.exists(adapter_path) else base
            _model.eval()
            print(f"[Node {NODE_ID}] Model loaded into cache.")

def _invalidate_model_cache():
    """Call after aggregation so next /predict picks up the merged adapter."""
    global _model, _tokenizer
    with _model_lock:
        _model     = None
        _tokenizer = None
    print(f"[Node {NODE_ID}] Model cache invalidated — will reload on next /predict.")

# ── App 1: Local IPC (Java ↔ Python on 127.0.0.1) ────────────────────────────
local_app = Flask("local")

@local_app.route("/train", methods=["POST"])
def start_train():
    data = request.get_json(silent=True) or {}
    round_num = data.get("round", 1)
    _set_status("training", round_num)
    threading.Thread(target=_do_train, args=(round_num,), daemon=True).start()
    return jsonify({"status": "training_started", "round": round_num})

def _do_train(round_num):
    try:
        _lt.train(round_num)
        _set_status("training_complete", round_num)
    except Exception as e:
        print(f"[Node {NODE_ID}] Training error: {e}")
        _set_status("training_error", round_num)

@local_app.route("/status", methods=["GET"])
def get_status():
    return jsonify(_get_status())

@local_app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    symptoms = data.get("symptoms", "")
    result = _run_inference(symptoms)
    return jsonify({"diagnosis": result})

@local_app.route("/aggregate", methods=["POST"])
def aggregate():
    data = request.get_json(silent=True) or {}
    round_num = data.get("round", 1)
    # FIX (Issue 2 / Bug 7): peers map comes from Java and already excludes self
    # (RoundController.buildPeersJson now filters out nodeParam.getNodeId())
    peers = data.get("peers", {})
    peer_map = {int(k): v for k, v in peers.items()}
    _set_status("aggregating", round_num)
    threading.Thread(target=_do_aggregate, args=(peer_map, round_num), daemon=True).start()
    return jsonify({"status": "aggregation_started"})

def _do_aggregate(peer_map, round_num):
    try:
        # Pass self_node_id so aggregator loads own adapter from disk, not via HTTP
        _agg.collect_and_aggregate(peer_map, round_num, OUTPUT_DIR, self_node_id=NODE_ID)
        _invalidate_model_cache()  # force model reload on next /predict
        _set_status("aggregate_complete", round_num)
    except Exception as e:
        print(f"[Node {NODE_ID}] Aggregation error: {e}")
        _set_status("aggregate_error", round_num)

# ── App 2: Weight Transfer Server (peers pull from this on LAN) ───────────────
weight_app = Flask("weights")

@weight_app.route("/weights", methods=["GET"])
def serve_weights():
    path = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
    if not os.path.exists(path):
        return jsonify({"error": "No adapter available yet"}), 404
    return send_file(
        os.path.abspath(path),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"adapter_node_{NODE_ID}.safetensors"
    )

# ── Inference helper ──────────────────────────────────────────────────────────
def _run_inference(symptoms: str) -> str:
    _ensure_model_loaded()
    inputs = _tokenizer(f"Symptoms: {symptoms}\nDiagnosis:", return_tensors="pt")
    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=50, pad_token_id=_tokenizer.eos_token_id)
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[Node {NODE_ID}] IPC server  → 127.0.0.1:{LOCAL_PORT}")
    print(f"[Node {NODE_ID}] Weight server → 0.0.0.0:{WEIGHT_PORT}")

    threading.Thread(
        target=lambda: local_app.run(host="127.0.0.1", port=LOCAL_PORT, threaded=True),
        daemon=True
    ).start()

    # Weight server binds 0.0.0.0 so LAN peers can reach it
    weight_app.run(host="0.0.0.0", port=WEIGHT_PORT, threaded=True)