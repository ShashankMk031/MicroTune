# MicroTune Runbook For macOS

This guide covers the Mac-only workflow for MicroTune after training has already been completed elsewhere and the adapter has been pulled into `microtune_final/`.

## 1. What This Repo Contains

- Base model target in the project code: `google/gemma-2b`
- Fine-tuned LoRA adapter directory: `microtune_final/`
  - This repo assumes `microtune_final/` already contains the adapter exported from your Kaggle training run
- Dataset flow: GSM8K raw -> processed dataset in `datasets/gsm8k_processed`
- Serving stack:
  - FastAPI: [api/app.py](./api/app.py)
  - Gradio UI: [ui/app.py](./ui/app.py)
- Evaluation:
  - Accuracy script: [evaluation/eval.py](./evaluation/eval.py)
  - Side-by-side comparison: [evaluation/compare.py](./evaluation/compare.py)
- Merge/export:
  - Merge LoRA: [scripts/merge_lora.py](./scripts/merge_lora.py)
  - GGUF export notes: [scripts/export_gguf.md](./scripts/export_gguf.md)

## 2. macOS Prerequisites

Install Apple developer tools:

```bash
xcode-select --install
```

Install Homebrew if it is not already installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install core tools:

```bash
brew install python@3.12 git cmake
```

Optional but recommended:

```bash
brew install uv
```

Notes for Mac users:

- Apple Silicon is preferred over Intel for local inference.
- `bitsandbytes` is generally a Linux + NVIDIA path, not a Mac path.
- Local Mac usage is best for:
  - dataset preprocessing
  - evaluation and comparison
  - API and UI development
  - light inference if memory permits
  - GGUF-based local serving after exporting on a supported machine

## 3. Clone And Enter The Project

```bash
git clone https://github.com/ShashankMk031/MicroTune.git
cd MicroTune
```

## 4. Create A Virtual Environment

Using standard `venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Or with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install --upgrade pip
```

## 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

If you need the project metadata dependencies too:

```bash
pip install -e .
```

If `bitsandbytes` fails on Mac, remove it from the install command and continue:

```bash
pip install torch transformers datasets peft trl accelerate evaluate wandb fastapi uvicorn gradio matplotlib
```

## 6. Hugging Face Authentication

Gemma checkpoints may require Hugging Face access approval.

Log in:

```bash
huggingface-cli login
```

Or:

```bash
hf auth login
```

If the model download is gated, confirm your Hugging Face account has access to the Gemma checkpoint you intend to use.

## 7. Prepare The Dataset

Create the processed GSM8K dataset:

```bash
python datasets/preprocess.py
```

Expected output directory:

```text
datasets/gsm8k_processed/
```

This repo’s training and evaluation flow uses the processed dataset, not the tokenized dataset, by default.

## 8. Confirm The Fine-Tuned Adapter Exists

These local scripts assume the fine-tuned LoRA adapter already exists here:

```text
microtune_final/
```

You do not need to run [training/train.py](./training/train.py) on your Mac for the remaining steps in this guide.

## 9. Evaluate Base vs Fine-Tuned Model

Run exact-match accuracy on the processed test split locally on your MacBook:

```bash
python evaluation/eval.py
```

This script compares:

- base model output
- fine-tuned model output using the adapter in `microtune_final/`
- runtime device on macOS: `mps` if available, otherwise `cpu`

Useful flags:

```bash
python evaluation/eval.py --limit 100
python evaluation/eval.py --max-new-tokens 256
python evaluation/eval.py --dataset-path datasets/gsm8k_processed
python evaluation/eval.py --device cpu
```

Expected output format:

```text
Base Model Accuracy: X%
Fine-tuned Model Accuracy: Y%
Improvement: +Z%
```

Generated files:

```text
results/eval_metrics.json
results/accuracy_comparison.png
```

## 10. Compare Sample Outputs

Run side-by-side qualitative comparison locally on your MacBook:

```bash
python evaluation/compare.py
```

Examples:

```bash
python evaluation/compare.py --samples 10
python evaluation/compare.py --samples 15 --seed 7
python evaluation/compare.py --device cpu
```

Generated files:

```text
results/compare_samples.json
results/sample_match_comparison.png
```

## 11. Run The FastAPI Service

Start the API from the repository root using the module form:

```bash
python -m api.app
```

If you are not already inside the activated `.venv`:

```bash
source .venv/bin/activate
python -m api.app
```

The API will automatically detect your MacBook's MPS (Metal Performance Shaders) support and use it for inference. If MPS is not available, it will fall back to CPU.

By default it serves on:

```text
http://0.0.0.0:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Solve endpoint:

```bash
curl -X POST http://127.0.0.1:8000/solve \
  -H "Content-Type: application/json" \
  -d '{"question":"If a shop sells 3 notebooks for $12, how much do 5 notebooks cost?"}'
```

Environment variables:

```bash
export MICROTUNE_BASE_MODEL="google/gemma-2b"
export MICROTUNE_ADAPTER_PATH="microtune_final"
export MICROTUNE_MAX_NEW_TOKENS="256"
export MICROTUNE_API_HOST="0.0.0.0"
export MICROTUNE_API_PORT="8000"
```

## 12. Run The Gradio UI

If you want the UI to call the API:

```bash
python api/app.py
```

Open another terminal:

```bash
source .venv/bin/activate
python ui/app.py
```

If you want the UI to load the model directly instead of using the API:

```bash
export MICROTUNE_USE_DIRECT_MODEL=1
python ui/app.py
```

Default UI address:

```text
http://127.0.0.1:7860
```

## 13. Merge LoRA Into The Base Model

Run:

```bash
python scripts/merge_lora.py
```

This writes the merged checkpoint into:

```text
merged_model/
```

The merge script uses the adapter stored in `microtune_final/` by default.

You can also override paths:

```bash
python scripts/merge_lora.py \
  --base-model google/gemma-2b \
  --adapter-path microtune_final \
  --output-path merged_model
```

## 14. Export To GGUF

See the full notes in [scripts/export_gguf.md](./scripts/export_gguf.md).

High-level flow:

1. Merge LoRA into a full Hugging Face model directory.
2. On your Mac, build `llama.cpp` and run `convert_hf_to_gguf.py`.
3. Quantize the generated `.gguf` file if needed.
4. Load the GGUF artifact into LM Studio or another GGUF-compatible runtime on macOS.

## 15. Generated Graphs

After running the local evaluation scripts, these graph files should exist:

```text
results/accuracy_comparison.png
results/sample_match_comparison.png
```

The README embeds both of these generated images.

## 16. Deployment Options From A Mac Workflow

### Option A: Local API Deployment On Your Mac

Good for development or private local usage.

Run with `uvicorn` directly:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

This gives you a stable local HTTP service that the Gradio UI can consume.

### Option B: Local GGUF Deployment On macOS

Best option if you want reliable Mac inference without the Python Hugging Face stack.

Recommended flow:

1. Merge the adapter locally on your Mac.
2. Export the merged model to GGUF.
3. Open the GGUF model in LM Studio on your Mac.

This is the most practical deployment route for Apple Silicon laptops and desktops.

## 17. Common Errors And Fixes

### `ImportError` or `ModuleNotFoundError: No module named ...`

Cause:

- Virtual environment not activated
- dependency install did not complete

Fix:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### `bitsandbytes` installation fails on macOS

Cause:

- `bitsandbytes` is not a standard Mac dependency path

Fix:

- Skip it locally on Mac.
- Use Linux + NVIDIA only if you need 4-bit CUDA training/inference.

Install the rest manually:

```bash
pip install torch transformers datasets peft trl accelerate evaluate wandb fastapi uvicorn gradio
```

### Hugging Face 401, 403, or gated model access error

Cause:

- not logged in
- Gemma access not granted for your account

Fix:

```bash
huggingface-cli login
```

Then verify model access in your Hugging Face account.

### `OSError` when loading the base model

Cause:

- model name not accessible
- network issue during download
- insufficient disk space

Fix:

- verify login and model access
- retry on a stable connection
- free disk space
- if the adapter was trained against a different base model, let the script auto-resolve it from `microtune_final/adapter_config.json`

### `Dataset not found at datasets/gsm8k_processed`

Cause:

- preprocessing has not been run yet

Fix:

```bash
python datasets/preprocess.py
```

### `Processed dataset example does not match the expected prompt format.`

Cause:

- dataset format differs from the repo’s expected GSM8K prompt template

Fix:

- regenerate the dataset with [datasets/preprocess.py](./datasets/preprocess.py)
- or update evaluation/inference prompt parsing consistently across the repo

### Out-of-memory during inference

Cause:

- Gemma model too large for available RAM or VRAM

Fix:

- reduce generation length:

```bash
export MICROTUNE_MAX_NEW_TOKENS=128
```

- close other memory-heavy applications
- prefer API mode over multiple direct model loads
- on Mac, prefer GGUF + LM Studio if the Hugging Face path is too heavy

### `MPS is not available on this Mac. Use --device cpu instead.`

Cause:

- your Mac does not expose Apple Metal Performance Shaders to PyTorch
- common on Intel Macs or unsupported PyTorch builds

Fix:

```bash
python evaluation/eval.py --device cpu
python evaluation/compare.py --device cpu
```

### Graph file was not created in `results/`

Cause:

- evaluation script exited early
- `matplotlib` was not installed

Fix:

```bash
pip install matplotlib
python evaluation/eval.py --limit 20
python evaluation/compare.py --samples 10
```

### API starts, but `/solve` is very slow

Cause:

- model is running on CPU instead of MPS (Apple Silicon acceleration)

Fix:

- the API automatically detects and uses MPS when available
- if inference is still slow, ensure your Mac has Apple Silicon and MPS is working
- keep the process warm for better performance

### Gradio UI cannot reach the API

Cause:

- API is not running
- wrong port or host

Fix:

```bash
python api/app.py
python ui/app.py
```

Or point the UI at a custom API URL:

```bash
export MICROTUNE_API_URL="http://127.0.0.1:8000"
python ui/app.py
```

### `No module named datasets` even though `datasets/` folder exists

Cause:

- local repo folder name can shadow the Hugging Face `datasets` package in some scripts

Fix:

- use the repo scripts directly as written
- avoid renaming imports manually unless you understand the `sys.path` workaround already included in preprocessing and evaluation code

## 18. Suggested Realistic Mac Workflow

If you are developing from a Mac, this is the cleanest path:

1. Install Python dependencies locally.
2. Run `python datasets/preprocess.py`.
3. Reuse `microtune_final/` directly.
4. Run `python evaluation/eval.py` to generate metrics and the accuracy graph.
5. Run `python evaluation/compare.py --samples 10` to generate sample comparisons and the sample graph.
6. Use the Mac for API/UI development and smoke tests.
7. Merge the adapter.
8. Export GGUF.
9. Run the GGUF model on macOS with LM Studio for day-to-day local use.
