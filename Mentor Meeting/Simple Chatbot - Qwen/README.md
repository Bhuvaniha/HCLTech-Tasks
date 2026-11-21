# Simple Chatbot

A minimal terminal chatbot that runs `Qwen/Qwen2.5-1.5B-Instruct` locally with
`transformers`. No token or internet connection is needed after the model is
downloaded.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Optionally create an environment file. No token is required, and the model
   is set in `config.py`:

   ```bash
   cp .env.example .env
   ```

3. Run the chatbot:

   ```bash
   python main.py
   ```

The first run downloads the model (about 3 GB). Later runs load it from the
local Hugging Face cache and work offline.

## Commands

- `quit` or `exit` — leave the chatbot
- `clear` — reset the conversation history
- `Ctrl+C` — exit gracefully

## Configuration

`config.py` holds the settings:

- `MODEL_NAME` — defaults to `Qwen/Qwen2.5-1.5B-Instruct`
- `MAX_TOKENS=512`
- `TEMPERATURE=0.7`
- `MAX_HISTORY_TURNS=10`

The model runs on CPU by default. Plan for roughly 6–8 GB of free RAM with
default float32 loading; memory use is lower if loaded in reduced precision.
