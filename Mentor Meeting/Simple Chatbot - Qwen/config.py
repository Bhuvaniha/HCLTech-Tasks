"""Configuration values for the local chatbot."""

import os

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_TOKENS: int = 512
TEMPERATURE: float = 0.7
MAX_HISTORY_TURNS: int = 10
