from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# Setting the pad token to be the same as the eos token
tokenizer.pad_token = tokenizer.eos_token 

print(tokenizer.pad_token)
print(tokenizer.eos_token)

dataset = load_from_disk("datasets/gsm8k_processed")

# Tokenizing the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation = True, padding = "max_length", max_length = 512)

# Applying the tokenization function to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched = True, remove_columns=["text"])

# Decoding the first tokenized example to verify the tokenization
decoded = tokenizer.decode(tokenized_datasets["train"][0]["input_ids"], skip_special_tokens = True)
print(decoded[:500])

# Checking the column names and lengths of the tokenized datasets
print(tokenized_datasets["train"].column_names)
print(tokenized_datasets["test"].column_names)

print(len(tokenized_datasets["train"]))
print(len(tokenized_datasets["test"]))

# Saving the tokenized dataset to disk
tokenized_datasets.save_to_disk("datasets/gsm8k_tokenized")

# Loading the tokenized dataset to verify
loaded = load_from_disk("datasets/gsm8k_tokenized")
loaded["train"][0]
loaded["test"][0]
