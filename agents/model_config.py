import os
from functools import lru_cache

MODEL_PROFILES = {
    "fast": dict(
        repo_id="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        filename="qwen2.5-coder-1.5b-instruct-q5_k_m.gguf",
    ),
    "balanced": dict(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
    ),
    "quality": dict(
        repo_id="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        filename="qwen2.5-coder-14b-instruct-q4_k_m.gguf",
    ),
    "frontier": dict(
        repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        filename="qwen3-coder-30b-a3b-instruct-q4_k_m.gguf",
    ),
}

DEFAULT_TIER = os.environ.get("MODEL_TIER", "balanced")
N_CTX = int(os.environ.get("MODEL_N_CTX", 16384))
N_GPU_LAYERS = int(os.environ.get("MODEL_N_GPU_LAYERS", -1))  # -1 = offload everything if a GPU is available


class _MockLlm:
    """Stand-in for llama_cpp.Llama used when MOCK_LLM=1. Lets every
    agent's plumbing (prompt building, JSON read/write, heuristics)
    run and be checked without a real model in the loop."""

    def create_completion(self, prompt, max_tokens=512, temperature=0.7, **_):
        return {"choices": [{"text": "[MOCK RESPONSE -- set MOCK_LLM=0 for a real critique]"}]}


@lru_cache(maxsize=None)
def get_llm(tier: str = None):
    """Load (once) and return a shared llama_cpp.Llama instance for the
    requested tier. Cached so critic/refactor/reviewer all reuse one
    loaded model instead of loading it separately."""
    if os.environ.get("MOCK_LLM"):
        return _MockLlm()

    tier = tier or DEFAULT_TIER
    if tier not in MODEL_PROFILES:
        raise ValueError(f"Unknown model tier '{tier}'. Choose one of {list(MODEL_PROFILES)}")

    from llama_cpp import Llama  # imported lazily so MOCK_LLM=1 needs no GPU build of llama-cpp-python

    profile = MODEL_PROFILES[tier]
    return Llama.from_pretrained(
        repo_id=profile["repo_id"],
        filename=profile["filename"],
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
