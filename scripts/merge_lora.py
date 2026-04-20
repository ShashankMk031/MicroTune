import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = "google/gemma-2b"
DEFAULT_ADAPTER_PATH = str(REPO_ROOT / "microtune_final")
DEFAULT_OUTPUT_PATH = str(REPO_ROOT / "merged_model")


def get_runtime_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this Mac. Use --device cpu instead.")
        return torch.device("mps")

    if device == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {device}")


def get_torch_dtype(runtime_device: torch.device) -> torch.dtype:
    if runtime_device.type == "mps":
        return torch.float16
    return torch.float32


def resolve_base_model_name(requested_base_model: str, adapter_path: str) -> str:
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        return requested_base_model

    with adapter_config_path.open("r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)

    adapter_base_model = adapter_config.get("base_model_name_or_path")
    if adapter_base_model:
        return adapter_base_model

    return requested_base_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the MicroTune LoRA adapter into the base model locally on macOS."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = args.adapter_path
    output_path = Path(args.output_path)
    base_model_name = resolve_base_model_name(args.base_model, adapter_path)
    runtime_device = get_runtime_device(args.device)

    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"LoRA adapter path not found: {Path(adapter_path).resolve()}")

    model_kwargs = {
        "torch_dtype": get_torch_dtype(runtime_device),
        "low_cpu_mem_usage": True,
    }

    tokenizer_source = adapter_path if Path(adapter_path).exists() else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(runtime_device)
    merged_model = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print(f"Merged on device: {runtime_device.type}")
    print(f"Merged model saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
