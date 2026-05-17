from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, set_peft_model_state_dict
from datasets import load_dataset
from safetensors.torch import load_file
import torch, os


NODE_ID = 1
OUTPUT_DIR = f"output/p{NODE_ID}_gpt2_lora"
DATASET_PATH = f"dataset_part_{NODE_ID}.jsonl"


def _enable_lora_grads(model):
    """
    Re-enable gradients on all LoRA parameters.
    Required after PeftModel.from_pretrained() or set_peft_model_state_dict()
    because loading from disk marks tensors as non-leaf / requires_grad=False,
    which causes 'does not require grad and does not have a grad_fn' during training.
    """
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
    trainable = sum(p.requires_grad for p in model.parameters())
    print(f"[Node {NODE_ID}] Trainable parameters: {trainable}")


def train(round_num: int):
    tokenizer = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = GPT2LMHeadModel.from_pretrained("distilbert/distilgpt2")

    adapter_path = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")

    if round_num > 1 and os.path.exists(adapter_path):
        print(f"[Node {NODE_ID}] Round {round_num}: Loading merged adapter to continue training.")
        model = PeftModel.from_pretrained(base, OUTPUT_DIR, is_trainable=True)
        # is_trainable=True tells PEFT to keep requires_grad=True on LoRA params.
        # We still call _enable_lora_grads as a safety net in case PEFT misses any.
        _enable_lora_grads(model)
        model.train()
    else:
        print(f"[Node {NODE_ID}] Round {round_num}: Initializing fresh LoRA adapter.")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            target_modules=["c_attn"],
            lora_dropout=0.1,
            bias="none"
        )
        model = get_peft_model(base, lora_config)
        model.train()

    if not os.path.exists(DATASET_PATH):
        print(f"Warning: {DATASET_PATH} not found. Skipping training.")
        return

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    def tokenize(example):
        question = example.get("Question") or ""
        cot      = example.get("Complex_CoT") or ""
        response = example.get("Response") or ""
        text = f"Question: {question}\nReasoning: {cot}\nAnswer: {response}"

        result = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256
        )
        result["labels"] = list(result["input_ids"])
        return result

    dataset = dataset.map(
        tokenize,
        batched=False,
        remove_columns=["Question", "Complex_CoT", "Response"]
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        max_steps=3,        # Testing only — remove for real training
        save_strategy="no",
        learning_rate=2e-4,
        logging_steps=10,
        report_to="none",
        dataloader_pin_memory=False,   # CPU-only — pin_memory has no effect
        dataloader_num_workers=0,      # Avoid multiprocess memory overhead
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    print(f"[Node {NODE_ID}] Round {round_num} training complete. Adapter saved to {OUTPUT_DIR}")