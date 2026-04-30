# Meridian Electronics - Customer Support Bot

Customer support chatbot for Meridian Electronics, built with GPT-4o-mini, the OpenAI Agents SDK, and a live MCP backend.

## Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o-mini |
| Agent framework | openai-agents 0.14.8 |
| Tool protocol | MCP over Streamable HTTP |
| UI | Gradio 6.13.0 |
| Deployment | Hugging Face Spaces |

## What it does

- Authenticates customers by email and PIN before doing anything else
- Lets customers browse products by category or search by description
- Shows order history and per-order line items
- Places orders with a mandatory confirmation step and a live price lookup before each order

## Local setup

```bash
pip install -r requirements.txt

# Copy .env.example or just export directly:
export OPENAI_API_KEY=sk-...

python app.py
# opens at http://localhost:7860
```

## Deploying to Hugging Face Spaces

1. Create a new Space, select the Gradio SDK.
2. Push `app.py` and `requirements.txt`.
3. Add `OPENAI_API_KEY` under Settings > Secrets.
4. The Space runtime launches the app automatically.

## Architecture

The agent is stateless. On each turn, `result.to_input_list()` is passed back as input to the next `Runner.run()` call, which is how multi-turn conversation context is maintained without storing messages server-side.

Each browser tab gets its own `SessionState` via `gr.State`. There is no shared session dictionary.

The MCP connection is opened and closed on every chat turn rather than kept alive. This avoids stale connection issues on Cloud Run at the cost of a small amount of per-turn latency.

## Files

```
app.py            agent, MCP connection, Gradio UI
requirements.txt  pinned dependencies
README.md         this file
```
