import os
import argparse
import inspect
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import modeling_utils
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


MODEL_ID = "google/gemma-4-E4B-it"
RAW_DATA_PATH = "datasets/gsm8k_processed"
OUTPUT_DIR = "microtune_runs"
FINAL_DIR = "microtune_final"
MAX_LENGTH = 256
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
dataloader_pin_memory = True


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Faster GPU math on supported NVIDIA hardware.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

# Newer Transformers versions may try a large VRAM pre-allocation during
# `from_pretrained`, which is only a loading-speed optimization and can OOM on
# smaller GPUs. Disable it for this training script.
if hasattr(modeling_utils, "caching_allocator_warmup"):
    modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None


def get_training_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_and_tokenize_dataset(tokenizer: AutoTokenizer, data_path: str, max_length: int):
    dataset = load_from_disk(data_path)
    train_ds = dataset["train"]

    if "text" not in train_ds.column_names:
        raise ValueError(
            f"Expected a 'text' column in {data_path}, found {train_ds.column_names}."
        )

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    return train_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing training set",
    )


def find_lora_target_modules(model) -> list[str]:
    target_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    linear_class_names = {"Linear", "Linear4bit", "Linear8bitLt"}
    target_modules = set()

    for name, module in model.named_modules():
        leaf_name = name.rsplit(".", 1)[-1]

        # Gemma 4 E4B multimodal checkpoints contain vision/audio towers whose
        # projection names overlap with the language model. Restrict LoRA to the
        # text backbone only so PEFT does not try to wrap Gemma4ClippableLinear.
        if ".vision_tower." in name or ".audio_tower." in name:
            continue

        if ".language_model.layers." not in name and ".model.layers." not in name:
            continue

        is_supported_linear = (
            isinstance(module, torch.nn.Linear)
            or module.__class__.__name__ in linear_class_names
        )
        if not is_supported_linear:
            continue

        if leaf_name in target_suffixes:
            target_modules.add(name)
            continue

        if leaf_name == "linear":
            parent_name = name.rsplit(".", 1)[0]
            parent_leaf = parent_name.rsplit(".", 1)[-1]
            if parent_leaf in target_suffixes:
                target_modules.add(name)

    if not target_modules:
        raise RuntimeError(
            "Could not find supported text-only LoRA target modules in the loaded model."
        )

    return sorted(target_modules)


def main(resume: bool):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to fine-tune Gemma 4 E4B.")

    dtype = get_training_dtype()
    use_bf16 = dtype == torch.bfloat16
    use_fp16 = dtype == torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds = load_and_tokenize_dataset(
        tokenizer=tokenizer,
        data_path=RAW_DATA_PATH,
        max_length=MAX_LENGTH,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    target_modules = find_lora_target_modules(model)
    print(f"Found {len(target_modules)} LoRA target modules in the text backbone.")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_kwargs = {
        "output_dir": OUTPUT_DIR,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "warmup_steps": 50,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 2,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "optim": "paged_adamw_8bit",
        "report_to": "none",
        "seed": 42,
        "remove_unused_columns": False,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "save_safetensors": True,
    }

    training_arg_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in training_arg_params:
        training_kwargs["evaluation_strategy"] = "no"
    elif "eval_strategy" in training_arg_params:
        training_kwargs["eval_strategy"] = "no"

    supported_training_kwargs = {
        key: value
        for key, value in training_kwargs.items()
        if key in training_arg_params
    }
    skipped_training_kwargs = sorted(set(training_kwargs) - set(supported_training_kwargs))
    if skipped_training_kwargs:
        print(
            "Skipping unsupported TrainingArguments keys: "
            + ", ".join(skipped_training_kwargs)
        )

    training_args = TrainingArguments(**supported_training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    if resume:
        print("Resuming training from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Starting fresh training...")
        trainer.train()

    Path(FINAL_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    print(f"Training complete. Adapter saved to {FINAL_DIR}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    main(resume=args.resume)
