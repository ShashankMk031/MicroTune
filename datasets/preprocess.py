from datasets import load_dataset, DatasetDict, load_from_disk

# Load dataset (contains train + test)
ds = load_dataset("openai/gsm8k", "main")

# Formatting function
def format_example(example):
    question = example["question"]
    answer = example["answer"]

    formatted_eg = (
        "### Instruction:\n"
        "Solve the following math problem step by step.\n\n"
        "### Question:\n"
        f"{question.strip()}\n\n"
        "### Response:\n"
        f"{answer.strip()}\n"
    )

    return {"text": formatted_eg}


#  Process both splits
train_dataset = ds["train"].map(format_example)
train_dataset = train_dataset.remove_columns(["question", "answer"])

test_dataset = ds["test"].map(format_example)
test_dataset = test_dataset.remove_columns(["question", "answer"])


# Filter invalid rows
def is_valid(example):
    return example["text"] is not None and len(example["text"].strip()) > 0

train_dataset = train_dataset.filter(is_valid)
test_dataset = test_dataset.filter(is_valid)


#  Combine into DatasetDict
processed_dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

#Saving to disk
processed_dataset.save_to_disk("datasets/gsm8k_processed")


# Reload + verify
loaded = load_from_disk("datasets/gsm8k_processed")

print(loaded.keys())                 # ['train', 'test']
print(loaded["train"].column_names) # ['text']
print(loaded["test"].column_names)  # ['text']

print(loaded["train"][0])
print(loaded["test"][0])