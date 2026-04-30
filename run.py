import sys
import os
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── MUST be first — load .env before anything else ──
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"

# If user runs `python run.py` from a global env, re-exec with project venv.
if VENV_PYTHON.exists():
    current_python = Path(sys.executable).resolve()
    if current_python != VENV_PYTHON.resolve():
        print(f"Switching to project venv Python: {VENV_PYTHON}")
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve())])

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f".env loaded from: {env_path}")
    key = os.environ.get("MISTRAL_API_KEY", "")
    if key:
        print(f"MISTRAL_API_KEY found: {key[:8]}...")
    else:
        print("MISTRAL_API_KEY not in .env file")
else:
    print(f".env file not found at: {env_path}")
    print("   Create it with your Mistral API key")

sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

if __name__ == "__main__":
    print("\nStarting NyayaAI...\n")
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # False prevents double-loading issue
    )