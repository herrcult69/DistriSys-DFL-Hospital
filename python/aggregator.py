import os, requests, torch, shutil
from safetensors.torch import load_file, save_file


def collect_and_aggregate(peer_addresses: dict, round_num: int, output_dir: str, self_node_id: int = None) -> str:
    """
    peer_addresses: {node_id (int): "ip:port" (str)} — should NOT include self's address.
                    Self's adapter is loaded directly from disk.
    output_dir:     Local node's PEFT output directory to overwrite adapter.
    self_node_id:   The calling node's own ID (used to load local adapter from disk).
    Returns:        Path to merged adapter file.
    """
    work_dir = f"/tmp/fl_round_{round_num}_{os.getpid()}"
    os.makedirs(work_dir, exist_ok=True)

    state_dicts = {}

    # Load own adapter from disk directly — no HTTP to self
    if self_node_id is not None:
        local_path = os.path.join(output_dir, "adapter_model.safetensors")
        if os.path.exists(local_path):
            state_dicts[self_node_id] = load_file(local_path)
            print(f"[FedIT] Loaded own adapter from disk (Node {self_node_id})")
        else:
            print(f"[FedIT] Warning: Local adapter not found at {local_path}. Skipping self.")

    # Download adapters from peers (not self)
    for node_id, addr in peer_addresses.items():
        if node_id == self_node_id:
            continue
        save_path = os.path.join(work_dir, f"adapter_node_{node_id}.safetensors")
        print(f"[FedIT] Downloading from http://{addr}/weights")
        try:
            resp = requests.get(f"http://{addr}/weights", stream=True, timeout=120)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            state_dicts[node_id] = load_file(save_path)
            print(f"[FedIT] Collected adapter from Node {node_id}")
        except Exception as e:
            print(f"[FedIT] Failed to download from Node {node_id} @ {addr}: {e}")

    if not state_dicts:
        print("[FedIT] No state dicts collected. Aborting aggregation.")
        shutil.rmtree(work_dir, ignore_errors=True)
        return ""

    merged = _fedit_merge(state_dicts, rank=8)

    # Blanket contiguous pass — safetensors requires all tensors to be contiguous
    # SVD outputs and any sliced/viewed tensors will fail save_file without this
    merged = {k: v.contiguous() for k, v in merged.items()}

    os.makedirs("output", exist_ok=True)
    archive_path = f"output/merged_adapter_round_{round_num}_{os.getpid()}.safetensors"
    save_file(merged, archive_path)
    print(f"[FedIT] Archive copy saved → {archive_path}")

    # Atomic write — write to .tmp then os.replace() to avoid
    # a concurrent /weights GET reading a partially written file
    local_adapter_path = os.path.join(output_dir, "adapter_model.safetensors")
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = local_adapter_path + ".tmp"
    save_file(merged, tmp_path)
    os.replace(tmp_path, local_adapter_path)

    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"[FedIT] Round {round_num} merged adapter atomically saved → {local_adapter_path}")
    return local_adapter_path


def _fedit_merge(state_dicts: dict, rank: int) -> dict:
    """
    Correct FedAvg for LoRA: avg(B_i @ A_i) per layer, then SVD re-decompose.
    This avoids the mathematically incorrect naive approach of avg(A_i) / avg(B_i).
    """
    all_keys = list(next(iter(state_dicts.values())).keys())
    merged = {}
    processed_keys = set()

    lora_prefixes = set()
    for k in all_keys:
        if "lora_A" in k:
            lora_prefixes.add(k.replace(".lora_A.weight", ""))

    for prefix in lora_prefixes:
        key_A = f"{prefix}.lora_A.weight"
        key_B = f"{prefix}.lora_B.weight"

        if key_A in all_keys and key_B in all_keys:
            delta_sum = sum(
                state_dicts[n][key_B].float() @ state_dicts[n][key_A].float()
                for n in state_dicts
            )
            delta_avg = delta_sum / len(state_dicts)

            U, S, Vh = torch.linalg.svd(delta_avg, full_matrices=False)
            sqrt_S = torch.sqrt(S[:rank].clamp(min=1e-8))

            # .contiguous() is REQUIRED here — SVD returns views into strided memory.
            # safetensors will raise "non contiguous tensor" without it.
            # Cast back to float16 to match the original adapter dtype.
            merged[key_B] = (U[:, :rank] * sqrt_S).contiguous().to(torch.float16)
            merged[key_A] = (torch.diag(sqrt_S) @ Vh[:rank, :]).contiguous().to(torch.float16)
            processed_keys.update([key_A, key_B])

    # Non-LoRA keys (biases etc.): simple mean
    for k in all_keys:
        if k not in processed_keys:
            merged[k] = (
                sum(state_dicts[n][k].float() for n in state_dicts) / len(state_dicts)
            ).contiguous().to(torch.float16)

    return merged