import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
MAX_LENGTH = 384


# Faster GPU math on supported NVIDIA hardware.
torch.backends.cuda.matmul.allow_tf32 = True


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
        torch_dtype=dtype,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        warmup_steps=50,
        logging_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        evaluation_strategy="no",
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        seed=42,
        remove_unused_columns=False,
    )

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
