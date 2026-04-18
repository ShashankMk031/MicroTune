# Run Gemma 4 E4B On Lightning AI

This guide shows how to run the Hugging Face Gemma 4 E4B multimodal example on a Lightning AI Studio using the free monthly GPU credits.

## What You Need

- A Lightning AI account
- A Hugging Face account
- Access to `google/gemma-4-E4B-it` on Hugging Face if the repo prompts for it

Gemma 4 support was added to Hugging Face Transformers on `2026-04-01`, so use a recent `transformers` version.

## 1. Start A Lightning Studio

1. Sign in to Lightning AI.
2. Create a new Studio.
3. Pick a single-GPU machine from the options available to your free account.

Lightning’s current docs say free accounts get monthly credits for GPU compute, while multi-GPU and multi-node runs require Pro or Teams.

## 2. Bring This Repo Into The Studio

You can either:

- Clone the repo from GitHub inside the Studio terminal, or
- Upload the project files into the Studio

Example:

```bash
git clone <your-repo-url>
cd MicroTune
```

## 3. Install Dependencies

Open the Studio terminal and run:

```bash
python -m pip install -U pip
python -m pip install -U torch torchvision transformers accelerate sentencepiece huggingface_hub
```

If you also want the training stack from this repo later, run:

```bash
python -m pip install -r requirements.txt
python -m pip install -U transformers accelerate torchvision sentencepiece huggingface_hub
```

## 4. Authenticate To Hugging Face

If Gemma access is gated for your account, request access in your browser first on the model page:

- [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)

Then log in from the Lightning terminal:

```bash
hf auth login
```

## 5. Create A Smoke Test Script

Create a file named `gemma4_smoke.py` with this content:

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/gemma-4-E4B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID, padding_side="left")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {
                "type": "text",
                "text": "Describe this image in one short paragraph.",
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        cache_implementation="static",
    )

response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
print(response)
```

## 6. Run It

```bash
python gemma4_smoke.py
```

If everything is correct, the model will download and print a short description of the sample image.

## 7. If You Want To Use A Notebook Instead

Paste the same Python code into a notebook cell in the Studio and run it there.

## 8. Important Notes

- `AutoModelForImageTextToText` is the right class when you want image + text input.
- For text-only chat, Hugging Face’s Gemma 4 model card uses `AutoModelForCausalLM` instead.
- Free GPU availability on Lightning changes over time. `google/gemma-4-E4B-it` may still be too large for some smaller free GPUs.
- If you hit out-of-memory errors, try:
  - Reducing `max_new_tokens`
  - Restarting on a larger available single-GPU machine
  - Using the smaller `google/gemma-4-E2B-it` model
  - Using the text-only `AutoModelForCausalLM` path instead of the multimodal path

## 9. Running This Repo’s Training Script

This repo also has [training/train.py](/Users/shashankmk/Documents/Projects-Development/MicroTune/training/train.py:1). That is for fine-tuning, not for the simple Hugging Face inference example above.

If you want to try training in Lightning Studio after dependency install:

```bash
python training/train.py
```

Resume from checkpoint with:

```bash
python training/train.py --resume
```

Be aware that fine-tuning Gemma 4 E4B can still exceed the memory available on smaller free GPUs even with 4-bit QLoRA.

## Sources

- [Hugging Face Gemma 4 model docs](https://huggingface.co/docs/transformers/model_doc/gemma4)
- [Hugging Face Gemma 4 E4B model card](https://huggingface.co/google/gemma-4-E4B)
- [Lightning AI Studios docs](https://lightning.ai/docs/pytorch/latest/clouds/lightning_ai.html)
- [Hugging Face gated model access docs](https://huggingface.co/docs/hub/models-gated)
