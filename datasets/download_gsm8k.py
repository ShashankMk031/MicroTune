from datasets import load_dataset

# Loading the GSM8K dataset
ds = load_dataset("openai/gsm8k", "main")

print(ds["train"][0])