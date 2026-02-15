# Thoughtful AI Customer Support Agent

A simple production-style customer support AI agent for Thoughtful AI.

## Features

- Conversational interface using Streamlit.
- Hardcoded Q&A knowledge base for Thoughtful AI agents (EVA, CAM, PHIL, and benefits).
- Best-match retrieval using string similarity.
- Fallback response path:
  - Uses OpenAI chat completion when `OPENAI_API_KEY` is set.
  - Uses a safe generic assistant response when no API key is available.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
pytest -q
```

## Notes

- Set optional environment variables for live LLM fallback:
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL` (defaults to `gpt-4o-mini`)
