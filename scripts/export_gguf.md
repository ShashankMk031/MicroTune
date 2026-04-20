# Export MicroTune to GGUF

## 1. Merge the LoRA adapter into the base model

Run the merge script first so `llama.cpp` sees a standard Hugging Face model directory:

```bash
python scripts/merge_lora.py
```

This writes the merged checkpoint to `merged_model/`.

## 2. Build `llama.cpp`

Clone and build the latest `llama.cpp` tools:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

## 3. Convert the merged Hugging Face model to GGUF

From the `llama.cpp` directory, run:

```bash
python convert_hf_to_gguf.py ../MicroTune/merged_model \
  --outfile ../MicroTune/merged_model/microtune-f16.gguf \
  --outtype f16
```

## 4. Optional: quantize for LM Studio or `llama.cpp`

Use the generated FP16 file as input to the quantizer:

```bash
./build/bin/llama-quantize \
  ../MicroTune/merged_model/microtune-f16.gguf \
  ../MicroTune/merged_model/microtune-q4_k_m.gguf \
  Q4_K_M
```

`microtune-q4_k_m.gguf` is a practical default for local inference in LM Studio.
