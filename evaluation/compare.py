import argparse
import gc
import json
import os
import random
import re
import sys
from pathlib import Path

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
DEFAULT_REPORT_PATH = str(RESULTS_DIR / "compare_samples.json")
DEFAULT_PLOT_PATH = str(RESULTS_DIR / "sample_match_comparison.png")


def import_hf_datasets():
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

    return final_segment.strip().splitlines()[0].strip()


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


def load_questions(dataset_path: str, split: str, sample_count: int, seed: int) -> list[tuple[str, str]]:
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

    rng = random.Random(seed)
    indices = list(range(len(split_dataset)))
    rng.shuffle(indices)
    sampled_indices = indices[: min(sample_count, len(indices))]

    samples = []
    for index in sampled_indices:
        question, reference_answer = parse_processed_example(split_dataset[index]["text"])
        samples.append((question, reference_answer))
    return samples


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


def run_model(
    base_model_name: str,
    runtime_device: torch.device,
    adapter_path: str | None,
    questions: list[tuple[str, str]],
    max_new_tokens: int,
) -> list[str]:
    model, tokenizer = load_model_and_tokenizer(
        base_model_name=base_model_name,
        runtime_device=runtime_device,
        adapter_path=adapter_path,
    )

    outputs = []
    for question, _ in questions:
        outputs.append(
            generate_response(
                model=model,
                tokenizer=tokenizer,
                runtime_device=runtime_device,
                prompt=build_prompt(question),
                max_new_tokens=max_new_tokens,
            )
        )

    del model
    gc.collect()
    clear_device_cache(runtime_device)

    return outputs


def save_report(report_path: str, plot_path: str, report: dict) -> None:
    report["plot_path"] = str(Path(plot_path).resolve())
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")


def save_sample_match_plot(plot_path: str, base_matches: int, fine_tuned_matches: int, total_samples: int) -> None:
    plot_file = Path(plot_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Base", "Fine-tuned"]
    values = [base_matches, fine_tuned_matches]
    colors = ["#8E9AAF", "#2A9D8F"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("Exact Matches On Sampled Questions")
    plt.ylim(0, max(total_samples, max(values) + 1))
    plt.title("MicroTune Sample Comparison")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.1,
            f"{value}/{total_samples}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base and fine-tuned MicroTune outputs locally on macOS."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
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
    samples = load_questions(
        dataset_path=args.dataset_path,
        split=args.split,
        sample_count=max(1, min(args.samples, 20)),
        seed=args.seed,
    )

    base_outputs = run_model(
        base_model_name=resolved_base_model,
        runtime_device=runtime_device,
        adapter_path=None,
        questions=samples,
        max_new_tokens=args.max_new_tokens,
    )
    fine_tuned_outputs = run_model(
        base_model_name=resolved_base_model,
        runtime_device=runtime_device,
        adapter_path=args.adapter_path,
        questions=samples,
        max_new_tokens=args.max_new_tokens,
    )

    comparisons = []
    base_matches = 0
    fine_tuned_matches = 0

    for index, ((question, reference_answer), base_output, fine_tuned_output) in enumerate(
        zip(samples, base_outputs, fine_tuned_outputs),
        start=1,
    ):
        base_final_answer = extract_final_answer(base_output)
        fine_tuned_final_answer = extract_final_answer(fine_tuned_output)
        reference_final_answer = extract_final_answer(reference_answer)

        base_match = normalize_answer(base_output) == normalize_answer(reference_answer)
        fine_tuned_match = normalize_answer(fine_tuned_output) == normalize_answer(reference_answer)
        base_matches += int(base_match)
        fine_tuned_matches += int(fine_tuned_match)

        comparisons.append(
            {
                "sample_index": index,
                "question": question,
                "reference_answer": reference_answer,
                "reference_final_answer": reference_final_answer,
                "base_output": base_output,
                "base_final_answer": base_final_answer,
                "base_match": base_match,
                "fine_tuned_output": fine_tuned_output,
                "fine_tuned_final_answer": fine_tuned_final_answer,
                "fine_tuned_match": fine_tuned_match,
            }
        )

        print(f"=== Sample {index} ===")
        print("Question:")
        print(question)
        print()
        print("Base Output:")
        print(base_output)
        print()
        print("Fine-tuned Output:")
        print(fine_tuned_output)
        print()
        print("Reference Answer:")
        print(reference_answer)
        print()

    save_report(
        report_path=args.report_path,
        plot_path=args.plot_path,
        report={
            "runtime_device": runtime_device.type,
            "base_model": resolved_base_model,
            "adapter_path": args.adapter_path,
            "split": args.split,
            "sample_count": len(comparisons),
            "base_exact_matches": base_matches,
            "fine_tuned_exact_matches": fine_tuned_matches,
            "comparisons": comparisons,
        },
    )
    save_sample_match_plot(
        plot_path=args.plot_path,
        base_matches=base_matches,
        fine_tuned_matches=fine_tuned_matches,
        total_samples=len(comparisons),
    )

    print(f"Running locally on device: {runtime_device.type}")
    print(f"Saved comparison report to: {Path(args.report_path).resolve()}")
    print(f"Saved comparison graph to: {Path(args.plot_path).resolve()}")


if __name__ == "__main__":
    main()
