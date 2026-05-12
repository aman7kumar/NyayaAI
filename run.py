"""
run.py — NyayaAI entry point.

CRITICAL: os.environ assignments for OpenMP/thread control MUST be the
very first statements in the process — before ANY library import.
Once libiomp5md.dll (Intel OpenMP, from PyTorch) or libomp.dll (LLVM
OpenMP, from PaddlePaddle) is loaded into the process, setting these
vars has no effect.  Both are loaded transitively by FastAPI startup,
so they must be set here, before `import uvicorn`.
"""

# ── Step 0: OpenMP / threading env vars — MUST be before ALL imports ─────────
import os

os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"   # allow both OpenMP runtimes
os.environ["OMP_NUM_THREADS"]        = "1"       # prevent thread-pool fights
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
# Paddle-specific: disable MKLDNN which adds its own OpenMP dependency
os.environ["FLAGS_use_mkldnn"]       = "0"
# Prevent PaddlePaddle from spawning worker processes
os.environ["PADDLE_TRAINERS_NUM"]    = "1"

# ── Step 1: Now safe to import everything else ────────────────────────────────
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f".env loaded from: {env_path}")

api_key = os.getenv("MISTRAL_API_KEY", "")
if api_key:
    print(f"MISTRAL_API_KEY found: {api_key[:8]}...")

import uvicorn

if __name__ == "__main__":
    print("\nStarting NyayaAI...\n")
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,       # reload=True re-imports modules and loses env vars
        workers=1,          # single worker — PaddleOCR is not fork-safe
        log_level="info",
    )