"""Load and cache the local text-generation pipeline."""

from functools import lru_cache

from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

from config import MODEL_NAME


def _load(local_files_only: bool) -> Pipeline:
    """Build the pipeline, optionally restricted to the local cache."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, local_files_only=local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, local_files_only=local_files_only
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    """Load the model once and return the cached text-generation pipeline.

    Use the local Hugging Face cache first so the chatbot runs fully offline
    after the initial download. Fall back to downloading only when the model
    has not been cached yet.
    """
    try:
        return _load(local_files_only=True)
    except OSError:
        return _load(local_files_only=False)
