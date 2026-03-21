from datasets import load_from_disk
from transformers import AutoTokenizer

dataset = load_from_disk("datasets/gsm8k_processed")

print("Dataset loaded:")
print(dataset)

# Initialize tokenizer
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad token to eos , since Mistral doesn't have a dedicated pad token give None to avoid errors during tokenization. This is common for causal language models where the EOS token is used for padding.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Pad token:", tokenizer.pad_token)
print("EOS token:", tokenizer.eos_token)

# Define tokenization function
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# apply tokenization to the entire dataset
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Display tokenized dataset info and verify correctness
print("\nTokenized columns:")
print(tokenized_datasets["train"].column_names)

print("\nDataset sizes:")
print("Train:", len(tokenized_datasets["train"]))
print("Test:", len(tokenized_datasets["test"]))

# Decode a sample to verify correctness
decoded = tokenizer.decode(
    tokenized_datasets["train"][0]["input_ids"],
    skip_special_tokens=True
)

print("\nDecoded sample (first 500 chars):\n")
print(decoded[:500])

# Save the tokenized dataset to disk
save_path = "datasets/gsm8k_tokenized"
tokenized_datasets.save_to_disk(save_path)

print(f"\nTokenized dataset saved to: {save_path}")

# Last verification step: Load the saved dataset and check its contents
loaded = load_from_disk(save_path)

print("\nReload check:")
print("Train example keys:", loaded["train"][0].keys())
print("Test example keys:", loaded["test"][0].keys())