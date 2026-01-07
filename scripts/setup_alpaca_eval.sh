#!/bin/bash
"""
Setup script for AlpacaEval 2.0 integration
Installs dependencies and configures the environment
"""

echo "=== AlpacaEval 2.0 Setup ==="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using one:"
    echo "   python -m venv alpaca_eval_env"
    echo "   source alpaca_eval_env/bin/activate"
fi

# Install requirements
echo ""
echo "Installing dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    # Repo root doesn't have a requirements.txt; install the minimal eval deps.
    pip install -U alpaca-eval python-dotenv pandas tqdm transformers peft accelerate
fi

# Verify installation
echo ""
echo "Verifying alpaca-eval installation..."
if command -v alpaca_eval &> /dev/null; then
    echo "✓ alpaca-eval command available"
    alpaca_eval --version
else
    echo "❌ alpaca-eval command not found"
    echo "Trying direct installation..."
    pip install alpaca-eval
fi

# Check OpenAI API key
echo ""
echo "Checking OpenAI API configuration..."
if [[ -z "${OPENAI_API_KEY}" ]]; then
    echo "❌ OPENAI_API_KEY not set"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "   export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    echo "Add this to your shell profile (~/.bashrc, ~/.zshrc) to make it permanent"
else
    echo "✓ OPENAI_API_KEY is set"
fi

# Test the evaluation script
echo ""
echo "Testing evaluation script..."
if python -c "from eval import AlpacaEvaluator; print('✓ eval.py imports successfully')" 2>/dev/null; then
    echo "✓ New evaluation script is ready"
else
    echo "❌ Error importing eval.py"
    echo "Please check for any missing dependencies"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage Examples:"
echo "1. Evaluate a model:"
echo "   python eval.py --model_path /path/to/your/model"
echo ""
echo "2. Evaluate with limited samples:"
echo "   python eval.py --model_path /path/to/your/model --num_samples 50"
echo ""
echo "3. Generate-only (no OpenAI key required):"
echo "   python eval.py --model_path /path/to/your/model --num_samples 50 --generate_only"
echo ""
