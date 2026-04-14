import os
import torch 
import argparse

from datasets import load_from_disk 
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

def main(resume: bool) : # Type Annotation : eg: resume:bol | It is a way to specify a expected type of a variable, unction param or a return type 
    # Paths 
    data_path = "datasets/gsm8k_tokenized"
    output_dir = "/content/drive/MyDrive/microtune_runs"
    final_dir = "/content/drive/MyDrive/microtune_final"

    # Load dataset 
    dataset = load_from_disk(data_path) 
    train_ds = dataset['train']
    eval_ds = dataset['test'] 

    # Tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token 

    # 4 bit quantization config 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype = torch.float16, 
        bnb_4bit_use_double_quant= True, 
        bnb_4bit_quant_type= "nf4"
    )

    # Load QLoRA model 
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1", 
        quantization_config = bnb_config,
        device_map = "auto",
        trust_remote_code = True
    )

    # Prepare for kbit training 
    model = prepare_model_for_kbit_training(model) 

    # Enable gradient checkpointing and disable caching for training
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    # Lora config 
    lora_config = LoraConfig(
        r = 16, 
        lora_alpha = 32, 
        target_modules=['q_proj', 'v_proj'], 
        lora_dropout=0.05,
        bias = "none",
        task_type = "CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config) 

    # Print trainable param (sanity check) 
    model.print_trainable_parameters() 

    # Data collator for dynamic padding 
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer, 
        mlm = False
    )

    # Training arguments with resume capability 
    training_args = TrainingArguments(
        output_dir = output_dir, 

        per_device_train_batch_size= 1, 
        per_device_eval_batch_size = 1, 
        gradient_accumulation_steps = 8, 

        learning_rate = 2e-4,
        num_train_epochs = 3,

        warmup_steps=50, 

        logging_steps = 50, 

        save_strategy = "steps", 
        save_steps = 150, 
        save_total_limit=2, 

        evaluation_strategy = "steps",
        eval_steps = 200, 

        fp16 = True, 
        bf16=False, 

        report_to = None,
    )
    
    # Trainer 
    trainer = Trainer( 
        model = model, 
        args = training_args,
        train_dataset= train_ds,
        eval_dataset= eval_ds,
        data_collator=data_collator,
    )

    # Traini / Resume 
    if resume: 
        print("Resuming training from checkpoint...") 
        trainer.train(resume_from_checkpoint = True)
    else: 
        print("Starting fresh training... ")
        trainer.train()

    # Save final LoRA adapter 
    model.save_pretrained(final_dir) 
    tokenizer.save_pretrained(final_dir)

    print("Training complete. Model Saved") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args() 

    main(resume=args.resume)
