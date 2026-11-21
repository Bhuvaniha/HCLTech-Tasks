"""Chat orchestration: build messages, run the local model, record history."""

import torch
from transformers import Pipeline

from client import get_pipeline
from config import MAX_TOKENS, TEMPERATURE
from history import ChatHistory
from prompts import build_messages


class ChatService:
    """Handle a single chat turn with a locally loaded model."""

    def __init__(
        self,
        generator: Pipeline | None = None,
        history: ChatHistory | None = None,
    ) -> None:
        """Create the service with an optional pipeline and history store."""
        self.generator = generator
        self.history = history or ChatHistory()

    def get_reply(self, user_input: str) -> str:
        """Return the assistant reply, or a friendly error message."""
        messages = build_messages(self.history.get_messages(), user_input)

        try:
            if self.generator is None:
                self.generator = get_pipeline()
            generator = self.generator
        except (torch.cuda.OutOfMemoryError, MemoryError):
            return self._oom_message()
        except OSError as exc:
            return (
                "Error: failed to download or load the model "
                f"({exc}). Check your internet connection on the first run."
            )
        except Exception as exc:
            return f"Error: failed to load the model ({exc}). Check MODEL_NAME in config.py."

        try:
            outputs = generator(
                messages,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
            )
        except (torch.cuda.OutOfMemoryError, MemoryError):
            return self._oom_message()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                return self._oom_message()
            return f"Error: the model failed to generate a reply ({exc})."
        except Exception as exc:
            return f"Error: the model failed to generate a reply ({exc})."

        generated = outputs[0]["generated_text"]
        reply = generated[-1]["content"] if isinstance(generated, list) else generated
        if not reply:
            return "Error: the model returned an empty response."

        self.history.add_user(user_input)
        self.history.add_assistant(reply)
        return reply

    @staticmethod
    def _oom_message() -> str:
        """Return a friendly out-of-memory message."""
        return (
            "Error: out of memory while loading or running the model. "
            "Try closing other applications or use a smaller model."
        )
