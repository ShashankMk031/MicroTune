from datasets import load_from_disk 
from transformers import AutoTokenizer 

# Load the preprocessed dataset from disk
dataset = load_from_disk("datasets/gsm8k_processed")

# Load tokenizer (using base model for training)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Fix pad token 
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    tokenized = tokenizer(
        examples['text'],
        truncation = True,
        padding = False, # dynamic padding later 
        max_length = 512
    )

    # Labels for causal LM 
    tokenized['labels'] =  tokenized['input_ids'].copy()

    return tokenized

# Apply tokenization 
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched = True, 
    remove_columns = ['text']
)

# Save tokenized dataset 
tokenized_dataset.save_to_disk("datasets/gsm8k_tokenized") 


# Verification 
print("Tokenization complete")  
print(tokenized_dataset['train'][0].keys()) 
