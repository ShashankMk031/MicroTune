import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "google/gemma-2b"
DEFAULT_ADAPTER_PATH = "microtune_final"
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
    
    # Check for available devices
    if torch.cuda.is_available():
        device = "cuda"
        attn_implementation = "sdpa"
    elif torch.backends.mps.is_available():
        device = "mps"
        attn_implementation = None  # MPS doesn't support SDPA yet
        
        # Patch the caching allocator warmup to avoid memory issues on MPS
        import transformers.modeling_utils
        original_caching_allocator_warmup = transformers.modeling_utils.caching_allocator_warmup
        
        def patched_caching_allocator_warmup(*args, **kwargs):
            # Skip the warmup to avoid memory allocation issues
            return
        
        transformers.modeling_utils.caching_allocator_warmup = patched_caching_allocator_warmup
    else:
        device = "cpu"
        attn_implementation = None
        
        # Patch the caching allocator warmup to avoid memory issues on CPU
        import transformers.modeling_utils
        original_caching_allocator_warmup = transformers.modeling_utils.caching_allocator_warmup
        
        def patched_caching_allocator_warmup(*args, **kwargs):
            # Skip the warmup to avoid memory allocation issues
            return
        
        transformers.modeling_utils.caching_allocator_warmup = patched_caching_allocator_warmup
    
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    
    if device != "cpu":
        model_kwargs["device_map"] = "auto"
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
    else:
        # For CPU, use 4-bit quantization
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device"] = device

    tokenizer_source = adapter_path if Path(adapter_path).exists() else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    # For CPU and MPS, disable device mapping in PEFT to avoid accelerate issues
    if device in ["cpu", "mps"]:
        # Load PEFT config and create PeftModel first
        from peft import PeftConfig
        peft_config = PeftConfig.from_pretrained(adapter_path)
        from peft import PeftModel
        model = PeftModel(model, peft_config)
        
        # Ensure model is on the correct device before loading weights
        model = model.to(device)
        
        # Now load the adapter weights
        from peft import set_peft_model_state_dict, load_peft_weights
        adapter_weights = load_peft_weights(adapter_path)
        set_peft_model_state_dict(model, adapter_weights)
    else:
        from peft import PeftModel
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


class SolveRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Math reasoning question to solve.")


class SolveResponse(BaseModel):
    response: str
    final_answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    adapter_path = os.getenv("MICROTUNE_ADAPTER_PATH", DEFAULT_ADAPTER_PATH)
    requested_base_model = os.getenv("MICROTUNE_BASE_MODEL", DEFAULT_BASE_MODEL)
    resolved_base_model = resolve_base_model_name(requested_base_model, adapter_path)
    app.state.bundle = load_model_bundle(
        base_model_name=resolved_base_model,
        adapter_path=adapter_path,
    )
    app.state.max_new_tokens = int(
        os.getenv("MICROTUNE_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS))
    )
    yield
    del app.state.bundle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="MicroTune API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest) -> SolveResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    response, final_answer = generate_solution(
        bundle=app.state.bundle,
        question=question,
        max_new_tokens=app.state.max_new_tokens,
    )
    return SolveResponse(response=response, final_answer=final_answer)


if __name__ == "__main__":
    import uvicorn

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    uvicorn.run(
        "api.app:app",
        host=os.getenv("MICROTUNE_API_HOST", "0.0.0.0"),
        port=int(os.getenv("MICROTUNE_API_PORT", "8000")),
        reload=False,
    )
