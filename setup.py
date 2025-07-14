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

def setup():
    """Install deps and setup wandb if in Colab"""
    
    # Load .env first
    load_env()
    
    # Check if Colab
    try:
        import google.colab
        print("Colab detected - installing dependencies...")
        
        # Install packages
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "torch", "transformers", "peft", "accelerate", "trl", 
                       "jsonlines", "datasets", "wandb", "python-dotenv", "-q"])
        
        # Setup wandb
        try:
            import wandb
            # Try .env file first
            api_key = os.environ.get('WANDB_API_KEY')
            if api_key:
                wandb.login(key=api_key)
                print("✓ wandb logged in via .env file")
            else:
                # Try Colab secrets
                try:
                    from google.colab import userdata
                    api_key = userdata.get('WANDB_API_KEY')
                    wandb.login(key=api_key)
                    print("✓ wandb logged in via Colab secrets")
                except:
                    # Fall back to interactive
                    wandb.login()
                    print("✓ wandb logged in")
        except:
            print("⚠ wandb login failed - continuing without logging")
        
        # Set env vars
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        print("✓ Setup complete")
        
    except ImportError:
        # Not in Colab, just load .env
        load_env()

if __name__ == "__main__":
    setup()


