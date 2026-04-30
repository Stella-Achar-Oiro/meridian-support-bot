"""
Meridian Electronics customer support chatbot.

Stack: GPT-4o-mini, OpenAI Agents SDK, MCP (Streamable HTTP), Gradio.

The agent is stateless. Multi-turn context works by passing result.to_input_list()
back as input on the next call to Runner.run(). Per-session state (customer_id,
auth flag, conversation history) lives in gr.State, one instance per browser tab.

OPENAI_API_KEY must be set in the environment or in a .env file.

Search for [REVIEW] before shipping. Each flag notes what to verify.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import gradio as gr
from agents import Agent, Runner, RunConfig
from agents.mcp import MCPServerStreamableHttp
load_dotenv()


# [REVIEW] Secrets: OPENAI_API_KEY must be set before launch.
# On Hugging Face Spaces, add it under Settings > Secrets. Never hard-code it.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Export it or add it to your .env file."
    )

MCP_SERVER_URL = "https://order-mcp-74afyau24q-uc.a.run.app/mcp"

# [REVIEW] System prompt: this is the main safety surface. Read every line.
# Key invariants:
#   1. verify_customer_pin must be called before any other tool.
#   2. get_product must be called before create_order to get a live price.
#   3. unit_price must be a decimal string ("499.99"), not a float.
#   4. The agent must get explicit confirmation before placing any order.
#   5. The agent must never return one customer's data to another.
SYSTEM_PROMPT = """You are Meridian Support, the customer service assistant for Meridian Electronics.

AUTHENTICATION
Before doing anything else, authenticate the customer:
1. Ask for their email address and 4-digit PIN.
2. Call verify_customer_pin(email=<email>, pin=<pin>).
3. If it succeeds, greet them by name and continue.
4. If it fails, apologise and ask them to try again. Do not proceed.

Do not call any other tool until verify_customer_pin succeeds.

PRODUCTS
- List products: call list_products(category=<cat>) or list_products() for all.
- Search: call search_products(query=<term>) for natural-language queries.
- Details: call get_product(sku=<sku>) for full specs and the current price.

ORDERS
Viewing history:
  - Call list_orders(customer_id=<id>) to show the customer's orders.
  - Call get_order(order_id=<id>) for line-item detail on a single order.

Placing an order (always follow this sequence):
  Step 1. Call get_product(sku=<sku>) to get the current unit_price.
          Never use a price from a previous search result.
  Step 2. Show the customer a clear summary: product name, SKU, quantity, unit price, total.
          Ask: "Shall I place this order? (yes/no)"
  Step 3. Only after the customer says YES, call create_order with:
            customer_id : UUID from verify_customer_pin
            items       : list of objects, each with:
              sku        : string
              quantity   : integer greater than 0
              unit_price : decimal string, e.g. "499.99" not 499.99
              currency   : "USD"

If the customer says anything other than a clear YES, do not place the order.

RULES
- Only use the customer_id returned by verify_customer_pin for this session.
- Never show order or account data belonging to a different customer.
- If a request is ambiguous, ask a clarifying question rather than guess.
- Never repeat a PIN back in conversation.
"""


@dataclass
class SessionState:
    authenticated: bool = False
    customer_id: str | None = None
    customer_name: str | None = None
    input_list: list[Any] = field(default_factory=list)


def make_mcp_server() -> MCPServerStreamableHttp:
    # [REVIEW] Connection lifecycle: a new connection is opened and closed on
    # every chat turn. This is safe for a stateless HTTP MCP server but adds
    # roughly 100ms per turn. If latency becomes a problem, move to a
    # connection pool rather than a single shared connection (Cloud Run closes
    # idle connections).
    return MCPServerStreamableHttp(
        params={"url": MCP_SERVER_URL},
        cache_tools_list=True,
    )


def make_agent(mcp_server: MCPServerStreamableHttp) -> Agent:
    # [REVIEW] Model: gpt-4o-mini is required by the brief (cost constraint).
    # Swap for gpt-4o here if tool-chaining reliability becomes an issue.
    return Agent(
        name="MeridianSupport",
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT,
        mcp_servers=[mcp_server],
    )


async def _chat_turn(
    user_message: str,
    session: SessionState,
) -> tuple[str, SessionState]:
    mcp_server = make_mcp_server()

    async with mcp_server:
        agent = make_agent(mcp_server)

        if session.input_list:
            run_input = session.input_list + \
                [{"role": "user", "content": user_message}]
        else:
            run_input = user_message

        # [REVIEW] max_turns=10 covers the deepest real flow (auth + product
        # lookup + order confirmation + create). Raise if flows time out; lower
        # to cap costs.
        result = await Runner.run(
            agent,
            run_input,
            run_config=RunConfig(tracing_disabled=True),
            max_turns=10,
        )

    session.input_list = result.to_input_list()
    return result.final_output or "(no response)", session


def create_chat_fn():
    def chat(
        message: str,
        history: list[dict],
        session_state: SessionState,
    ) -> tuple[str, SessionState]:
        # [REVIEW] Event loop: Gradio runs handlers in a thread pool, so
        # asyncio.run() is correct here. If you move to a shared long-lived
        # MCP connection you will need a different loop strategy.
        return asyncio.run(_chat_turn(message, session_state))

    return chat


_static = Path(__file__).parent / "static"
HEADER_HTML = (_static / "header.html").read_text()
FOOTER_HTML = (_static / "footer.html").read_text()
CUSTOM_CSS  = (_static / "style.css").read_text()


def build_app() -> gr.Blocks:
    chat_fn = create_chat_fn()

    with gr.Blocks(title="Meridian Electronics Support", css=CUSTOM_CSS) as app:
        gr.HTML(HEADER_HTML)

        # lambda ensures each browser tab gets its own SessionState instance.
        session_state = gr.State(lambda: SessionState())

        # [REVIEW] additional_inputs/additional_outputs wire our SessionState
        # through Gradio's ChatInterface so state persists across turns without
        # a global dict.
        gr.ChatInterface(
            fn=chat_fn,
            additional_inputs=[session_state],
            additional_outputs=[session_state],
            chatbot=gr.Chatbot(
                label="Meridian Support",
                placeholder="Hello! Please share your email and 4-digit PIN to get started.",
                height=520,
                avatar_images=(
                    None,  # user: default
                    "https://api.dicebear.com/9.x/bottts-neutral/svg?seed=meridian&backgroundColor=2b6cb0",
                ),
            ),
            textbox=gr.Textbox(
                placeholder="Type your message…",
                container=False,
                scale=7,
            ),
            submit_btn="Send",
            examples=None,
            # [REVIEW] cache_examples=False: setting this True would pre-run
            # the agent without an authenticated session.
            cache_examples=False,
        )

        gr.HTML(FOOTER_HTML)

    return app


if __name__ == "__main__":
    # [REVIEW] On Hugging Face Spaces the runtime calls app.launch() itself;
    # this block is only used for local development.
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
