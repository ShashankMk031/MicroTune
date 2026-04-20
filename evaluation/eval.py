import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
import httpx
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


matplotlib.use("Agg")

import matplotlib.pyplot as plt


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_BASE_MODEL = "google/gemma-2b"
DEFAULT_ADAPTER_PATH = str(REPO_ROOT / "microtune_final")
DEFAULT_DATASET_PATH = str(REPO_ROOT / "datasets" / "gsm8k_processed")
DEFAULT_SPLIT = "test"
DEFAULT_REPORT_PATH = str(RESULTS_DIR / "eval_metrics.json")
DEFAULT_PLOT_PATH = str(RESULTS_DIR / "accuracy_comparison.png")


def import_hf_datasets():
    # The repo contains a local `datasets/` directory, so remove it from
    # `sys.path` before importing the Hugging Face package of the same name.
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.resolve()

    original_sys_path = list(sys.path)
    sys.path = [
        path
        for path in sys.path
        if Path(path or ".").resolve() not in {script_dir, repo_root}
    ]
    try:
        from datasets import load_from_disk
    finally:
        sys.path = original_sys_path

    return load_from_disk


load_from_disk = import_hf_datasets()


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


def clear_device_cache(runtime_device: torch.device) -> None:
    if runtime_device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


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


def build_prompt(question: str) -> str:
    return (
        "### Instruction:\n"
        "Solve the following math problem step by step.\n\n"
        "### Question:\n"
        f"{question.strip()}\n\n"
        "### Response:\n"
    )


def parse_processed_example(text: str) -> tuple[str, str]:
    question_marker = "### Question:\n"
    response_marker = "\n\n### Response:\n"

    if question_marker not in text or response_marker not in text:
        raise ValueError("Processed dataset example does not match the expected prompt format.")

    question_start = text.index(question_marker) + len(question_marker)
    response_start = text.index(response_marker)
    question = text[question_start:response_start].strip()
    answer = text[response_start + len(response_marker) :].strip()
    return question, answer


def extract_final_answer(text: str) -> str:
    if "####" in text:
        final_segment = text.rsplit("####", maxsplit=1)[-1]
    else:
        non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        final_segment = non_empty_lines[-1] if non_empty_lines else text

    lines = final_segment.strip().splitlines()
    return lines[0].strip() if lines else ""


def normalize_answer(text: str) -> str:
    cleaned = extract_final_answer(text)
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.rstrip(".")
    cleaned = " ".join(cleaned.split())

    # Prefer numeric normalization for GSM8K-style answers where the final
    # number is often wrapped in prose like "**Answer:** ... 142".
    number_matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if number_matches:
        number = number_matches[-1]
        if "." in number:
            try:
                as_float = float(number)
                if as_float.is_integer():
                    return str(int(as_float))
            except ValueError:
                pass
        return number

    return cleaned


def load_dataset_split(dataset_path: str, split: str, limit: int | None) -> Iterable[dict]:
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_root}. Expected a processed dataset saved to disk."
        )

    dataset = load_from_disk(str(dataset_root))
    if split not in dataset:
        raise KeyError(f"Split '{split}' not found. Available splits: {list(dataset.keys())}")

    split_dataset = dataset[split]
    if "text" not in split_dataset.column_names:
        raise ValueError(
            f"Split '{split}' must contain a 'text' column, found {split_dataset.column_names}."
        )

    if limit is not None:
        split_dataset = split_dataset.select(range(min(limit, len(split_dataset))))

    return split_dataset


def load_model_and_tokenizer(
    base_model_name: str,
    runtime_device: torch.device,
    adapter_path: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    torch_dtype = get_torch_dtype(runtime_device)
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }

    tokenizer_source = adapter_path if adapter_path and Path(adapter_path).exists() else base_model_name

    def _load_with_local_first(load_fn):
        try:
            # Prefer local cache/files first to avoid proxy/DNS failures when offline.
            return load_fn(local_files_only=True)
        except (OSError, httpx.HTTPError):
            try:
                return load_fn(local_files_only=False)
            except (OSError, httpx.HTTPError) as network_exc:
                has_proxy = bool(os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"))
                is_proxy_error = "proxy" in type(network_exc).__name__.lower()
                if not (has_proxy and is_proxy_error):
                    raise

                proxy_keys = [
                    "HTTPS_PROXY",
                    "HTTP_PROXY",
                    "ALL_PROXY",
                    "https_proxy",
                    "http_proxy",
                    "all_proxy",
                ]
                original_proxy_env = {key: os.environ.pop(key) for key in proxy_keys if key in os.environ}
                try:
                    return load_fn(local_files_only=False)
                finally:
                    for key, value in original_proxy_env.items():
                        os.environ[key] = value

    try:
        tokenizer = _load_with_local_first(
            lambda local_files_only: AutoTokenizer.from_pretrained(
                tokenizer_source, local_files_only=local_files_only
            ),
        )
    except (OSError, httpx.HTTPError) as exc:
        raise RuntimeError(
            f"Failed to load tokenizer from '{tokenizer_source}'. "
            "Tried local cache first, then network. "
            "If network is blocked (proxy/DNS), provide a local path with --base-model or pre-cache the model."
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        model = _load_with_local_first(
            lambda local_files_only: AutoModelForCausalLM.from_pretrained(
                base_model_name, local_files_only=local_files_only, **model_kwargs
            ),
        )
    except (OSError, httpx.HTTPError) as exc:
        raise RuntimeError(
            f"Failed to load model weights for '{base_model_name}'. "
            "Tried local cache first, then network. "
            "If network is blocked (proxy/DNS), provide a local path with --base-model or pre-cache the model."
        ) from exc

    if adapter_path:
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
        except (OSError, httpx.HTTPError) as exc:
            raise RuntimeError(
                f"Failed to load PEFT adapter from '{adapter_path}'. "
                "Make sure the adapter path exists and is accessible."
            ) from exc

    model.to(runtime_device)
    model.eval()
    return model, tokenizer


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    runtime_device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(runtime_device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_model(
    base_model_name: str,
    runtime_device: torch.device,
    adapter_path: str | None,
    dataset_path: str,
    split: str,
    max_new_tokens: int,
    limit: int | None,
) -> dict:
    model, tokenizer = load_model_and_tokenizer(
        base_model_name=base_model_name,
        runtime_device=runtime_device,
        adapter_path=adapter_path,
    )

    dataset = load_dataset_split(dataset_path=dataset_path, split=split, limit=limit)
    correct = 0
    total = 0

    for example in dataset:
        question, reference_answer = parse_processed_example(example["text"])
        prompt = build_prompt(question)
        prediction = generate_response(
            model=model,
            tokenizer=tokenizer,
            runtime_device=runtime_device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        normalized_prediction = normalize_answer(prediction)
        normalized_reference = normalize_answer(reference_answer)
        is_match = normalized_prediction == normalized_reference
        if is_match:
            correct += 1
        total += 1

    del model
    gc.collect()
    clear_device_cache(runtime_device)

    accuracy = correct / total if total else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }


def _relative_or_absolute(path: str) -> str:
    """Return relative path if possible, otherwise absolute path."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def save_report(
    report_path: str,
    plot_path: str,
    runtime_device: torch.device,
    split: str,
    base_model_name: str,
    adapter_path: str,
    base_metrics: dict,
    fine_tuned_metrics: dict,
) -> None:
    report = {
        "runtime_device": runtime_device.type,
        "split": split,
        "base_model": base_model_name,
        "adapter_path": _relative_or_absolute(adapter_path),
        "base_model_metrics": base_metrics,
        "fine_tuned_model_metrics": fine_tuned_metrics,
        "improvement_percent": (fine_tuned_metrics["accuracy"] - base_metrics["accuracy"]) * 100,
        "plot_path": _relative_or_absolute(plot_path),
    }

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")


def save_accuracy_plot(plot_path: str, base_percent: float, fine_tuned_percent: float) -> None:
    plot_file = Path(plot_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Base", "Fine-tuned"]
    values = [base_percent, fine_tuned_percent]
    colors = ["#8E9AAF", "#2A9D8F"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, max(100, max(values) + 5))
    plt.title("MicroTune Accuracy Comparison")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base and fine-tuned MicroTune models locally on macOS."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--plot-path", default=DEFAULT_PLOT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.adapter_path).exists():
        raise FileNotFoundError(
            f"Fine-tuned adapter folder not found at {Path(args.adapter_path).resolve()}."
        )

    runtime_device = get_runtime_device(args.device)
    resolved_base_model = resolve_base_model_name(args.base_model, args.adapter_path)

    base_metrics = evaluate_model(
        base_model_name=resolved_base_model,
        runtime_device=runtime_device,
        adapter_path=None,
        dataset_path=args.dataset_path,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    fine_tuned_metrics = evaluate_model(
        base_model_name=resolved_base_model,
        runtime_device=runtime_device,
        adapter_path=args.adapter_path,
        dataset_path=args.dataset_path,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )

    base_percent = base_metrics["accuracy"] * 100
    fine_tuned_percent = fine_tuned_metrics["accuracy"] * 100
    improvement = fine_tuned_percent - base_percent

    save_report(
        report_path=args.report_path,
        plot_path=args.plot_path,
        runtime_device=runtime_device,
        split=args.split,
        base_model_name=resolved_base_model,
        adapter_path=args.adapter_path,
        base_metrics=base_metrics,
        fine_tuned_metrics=fine_tuned_metrics,
    )
    save_accuracy_plot(
        plot_path=args.plot_path,
        base_percent=base_percent,
        fine_tuned_percent=fine_tuned_percent,
    )

    print(f"Running locally on device: {runtime_device.type}")
    print(f"Base Model Accuracy: {base_percent:.2f}%")
    print(f"Fine-tuned Model Accuracy: {fine_tuned_percent:.2f}%")
    print(f"Improvement: {improvement:+.2f}%")
    print(f"Saved report to: {Path(args.report_path).resolve()}")
    print(f"Saved graph to: {Path(args.plot_path).resolve()}")


if __name__ == "__main__":
    main()
