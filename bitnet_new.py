import os
import time

# Force MinGW and disable compilation optimizations
os.environ["TORCH_CUDA_ARCH_LIST"] = "All"
os.environ["CMAKE_CXX_COMPILER"] = r"C:\msys64\mingw64\bin\g++"
os.environ["CMAKE_C_COMPILER"] = r"C:\msys64\mingw64\bin\gcc"
os.environ["USE_NINJA"] = "0"
os.environ["CXX"] = r"C:\msys64\mingw64\bin\g++"
os.environ["CC"] = r"C:\msys64\mingw64\bin\gcc"
os.environ["TORCH_INDUCTOR_CXX_COMPILER"] = r"C:\msys64\mingw64\bin\g++"
os.environ["PYTORCH_FORCE_LEGACY_CODE"] = "1"

# Disable PyTorch JIT compilation
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCH_COMPILE_MODE"] = "reduce-overhead"

import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize FastAPI app
app = FastAPI(title="BitNet Model Server")

# Global model and tokenizer objects
model = None
tokenizer = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: Optional[int] = Field(default=5000)
    temperature: Optional[float] = Field(default=1.0)


class ChatResponse(BaseModel):
    response: str
    timing: Dict[str, float]
    metrics: Dict[str, float]


@app.on_event("startup")
async def startup_event():
    global model, tokenizer

    model_id = "microsoft/bitnet-b1.58-2B-4T"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    print("Model loaded and ready to serve!")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global model, tokenizer

    # Start timing the entire request
    start_time = time.time()

    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert to the format expected by the tokenizer
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    # Time the template application and tokenization
    template_start = time.time()
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokenize_time = time.time() - template_start

    # Count input tokens
    input_token_count = chat_input["input_ids"].shape[-1]

    # Time the actual generation
    generation_start = time.time()
    generation_config = {
        "max_new_tokens": request.max_new_tokens,
        "temperature": request.temperature,
    }

    with torch.no_grad():
        outputs = model.generate(**chat_input, **generation_config)
    generation_time = time.time() - generation_start

    # Count generated tokens (excluding input tokens)
    generated_token_count = outputs.shape[-1] - input_token_count

    # Time the decoding
    decode_start = time.time()
    response_text = tokenizer.decode(
        outputs[0][chat_input["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    decode_time = time.time() - decode_start

    # Calculate total time
    total_time = time.time() - start_time

    # Calculate performance metrics
    generation_tokens_per_sec = (
        generated_token_count / generation_time if generation_time > 0 else 0
    )
    total_tokens_per_sec = generated_token_count / total_time if total_time > 0 else 0

    # Create timing information
    timing = {
        "total_seconds": total_time,
        "tokenize_seconds": tokenize_time,
        "generation_seconds": generation_time,
        "decode_seconds": decode_time,
    }

    # Add comprehensive metrics
    metrics = {
        "input_tokens": input_token_count,
        "output_tokens": generated_token_count,
        "total_tokens": input_token_count + generated_token_count,
        "generation_tokens_per_second": generation_tokens_per_sec,
        "total_tokens_per_second": total_tokens_per_sec,
        "tokenize_tokens_per_second": (
            input_token_count / tokenize_time if tokenize_time > 0 else 0
        ),
        "decode_tokens_per_second": (
            generated_token_count / decode_time if decode_time > 0 else 0
        ),
    }

    return ChatResponse(response=response_text, timing=timing, metrics=metrics)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bitnet_new:app", host="0.0.0.0", port=8000, reload=False)
