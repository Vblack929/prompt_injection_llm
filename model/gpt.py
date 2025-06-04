import openai
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

class GPT:
    def __init__(self, model="gpt-3.5-turbo"):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.api_key is None:
            raise ValueError("OpenAI API key not found. Make sure it's set in your .env file or environment variables.")
        openai.api_key = self.api_key
        self.model = model

    def generate(self, prompt, max_tokens=150):
        """
        Generate text using the OpenAI API.

        Args:
            prompt (str): The input prompt for the model.
            max_tokens (int): The maximum number of tokens to generate.

        Returns:
            str: The generated text.
        """
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

# Example usage
if __name__ == "__main__":
    # The .env file should be in the same directory as this script when run directly,
    # or in the project root if run from there.
    try:
        gpt_model = GPT()
        example_prompt = "Translate the following English text to French: 'Hello, world!'"
        generated_text = gpt_model.generate(example_prompt)
        print(f"Prompt: {example_prompt}")
        print(f"Generated Text: {generated_text}")
    except ValueError as ve:
        print(ve)
    # Minimal error handling for unexpected issues
    except Exception as e:
        print(f"An unexpected error occurred during example usage: {e}") 