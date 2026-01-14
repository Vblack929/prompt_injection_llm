"""Simple setup for DPO experiments in Colab"""

import os
import sys
import subprocess

def load_env():
    """Load .env file if it exists"""
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        pass

def _disable_wandb_by_default():
    """
    Disable Weights & Biases logging/prompts by default.
    Users can override by exporting ENABLE_WANDB=1 (and setting WANDB_API_KEY).
    """
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")
    # HF Transformers respects this flag to avoid importing/initializing wandb integration.
    os.environ.setdefault("TRANSFORMERS_NO_WANDB", "1")

def _in_colab() -> bool:
    """
    Robust Colab detection.
    Importing `google.colab` can succeed in non-Colab environments if the package is installed,
    which can accidentally trigger interactive setup (like wandb login).
    """
    return bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") or os.environ.get("COLAB_BACKEND_VERSION"))

def setup():
    """Install deps and setup wandb if in Colab"""
    
    # Load .env first
    load_env()
    # Always disable wandb prompts by default (safe for local + Colab)
    _disable_wandb_by_default()
    
    # Check if Colab
    try:
        if not _in_colab():
            raise ImportError("Not in Colab")
        import google.colab  # noqa: F401
        print("Colab detected - installing dependencies...")
        
        # Install packages
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "torch", "transformers", "peft", "accelerate", "trl", 
                       "jsonlines", "datasets", "wandb", "python-dotenv", "-q"])
        
        # Optional wandb setup (explicit opt-in only; never interactive by default)
        if os.environ.get("ENABLE_WANDB", "0") == "1":
            try:
                import wandb
                api_key = os.environ.get("WANDB_API_KEY")
                if api_key:
                    wandb.login(key=api_key)
                    print("✓ wandb logged in via WANDB_API_KEY")
                else:
                    print("⚠ ENABLE_WANDB=1 but WANDB_API_KEY not set; skipping wandb login")
            except Exception:
                print("⚠ wandb setup failed - continuing without logging")
        
        # Set env vars
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        print("✓ Setup complete")
        
    except ImportError:
        # Not in Colab, just load .env
        load_env()
        _disable_wandb_by_default()

if __name__ == "__main__":
    setup()


