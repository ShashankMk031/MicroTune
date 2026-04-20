import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "google/gemma-2b"
DEFAULT_ADAPTER_PATH = "microtune_final"
DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_MAX_NEW_TOKENS = 256


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
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


def build_prompt(question: str) -> str:
    return (
        "### Instruction:\n"
        "Solve the following math problem step by step.\n\n"
        "### Question:\n"
        f"{question.strip()}\n\n"
        "### Response:\n"
    )


def extract_final_answer(text: str) -> str:
    if "####" in text:
        final_segment = text.rsplit("####", maxsplit=1)[-1]
    else:
        non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        final_segment = non_empty_lines[-1] if non_empty_lines else text

    return final_segment.strip().splitlines()[0].strip()


def load_model_bundle(base_model_name: str, adapter_path: str) -> dict:
    torch_dtype = get_torch_dtype()
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "sdpa"

    tokenizer_source = adapter_path if Path(adapter_path).exists() else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return {"model": model, "tokenizer": tokenizer}


def generate_solution(bundle: dict, question: str, max_new_tokens: int) -> tuple[str, str]:
    prompt = build_prompt(question)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]

    model_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

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
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    final_answer = extract_final_answer(response)
    return response, final_answer


USE_DIRECT_MODEL = os.getenv("MICROTUNE_USE_DIRECT_MODEL", "0") == "1"
API_URL = os.getenv("MICROTUNE_API_URL", DEFAULT_API_URL).rstrip("/")
MAX_NEW_TOKENS = int(os.getenv("MICROTUNE_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS)))
MODEL_BUNDLE = None

if USE_DIRECT_MODEL:
    adapter_path = os.getenv("MICROTUNE_ADAPTER_PATH", DEFAULT_ADAPTER_PATH)
    requested_base_model = os.getenv("MICROTUNE_BASE_MODEL", DEFAULT_BASE_MODEL)
    resolved_base_model = resolve_base_model_name(requested_base_model, adapter_path)
    MODEL_BUNDLE = load_model_bundle(
        base_model_name=resolved_base_model,
        adapter_path=adapter_path,
    )


def solve_with_api(question: str) -> tuple[str, str]:
    payload = json.dumps({"question": question}).encode("utf-8")
    request = urllib.request.Request(
        url=f"{API_URL}/solve",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))

    return body["response"], body["final_answer"]


def solve_question(question: str) -> tuple[str, str]:
    question = question.strip()
    if not question:
        return "", ""

    if USE_DIRECT_MODEL and MODEL_BUNDLE is not None:
        return generate_solution(MODEL_BUNDLE, question, MAX_NEW_TOKENS)

    try:
        return solve_with_api(question)
    except urllib.error.URLError as exc:
        error_message = (
            "Could not reach the MicroTune API. Start `api/app.py` or set "
            "`MICROTUNE_USE_DIRECT_MODEL=1` to run the UI without the API."
        )
        return error_message, f"{type(exc).__name__}: {exc}"


with gr.Blocks(title="MicroTune Solver") as demo:
    gr.Markdown("# MicroTune\nAsk a GSM8K-style math question and inspect the model reasoning.")

    question_input = gr.Textbox(
        label="Question",
        placeholder="If a train travels 60 miles in 1.5 hours, what is its average speed?",
        lines=4,
    )
    solve_button = gr.Button("Solve", variant="primary")
    reasoning_output = gr.Textbox(label="Full Reasoning", lines=14)
    final_answer_output = gr.Textbox(label="Final Answer", lines=2)

    solve_button.click(
        fn=solve_question,
        inputs=question_input,
        outputs=[reasoning_output, final_answer_output],
    )
    question_input.submit(
        fn=solve_question,
        inputs=question_input,
        outputs=[reasoning_output, final_answer_output],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name=os.getenv("MICROTUNE_UI_HOST", "0.0.0.0"),
        server_port=int(os.getenv("MICROTUNE_UI_PORT", "7860")),
    )
