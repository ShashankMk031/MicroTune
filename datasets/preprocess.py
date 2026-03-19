from datasets import load_dataset

# Loading the GSM8K dataset
ds = load_dataset("openai/gsm8k", "main")

# Extracts the question and answer from the dataset and formats it into a new structure | Alpaca instruction format
def format_example(example):
    question = example["question"]
    answer = example["answer"]

    formatted_eg = (
        "### Instruction:\n"
        "Solve the following math problem step by step.\n\n"
        "### Question:\n"
        f"{question.strip()}\n\n"
        "### Response:\n"
        f"{answer.strip()}"
    )

    return {"text": formatted_eg}

formatted_dataset= ds.map(format_example)
print(formatted_dataset["train"][0])