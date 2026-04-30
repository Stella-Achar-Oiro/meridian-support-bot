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
from agents.mcp import MCPServerStreamableHttp
from agents import Agent, Runner, RunConfig
import gradio as gr

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
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


HEADER_HTML = """
<div style="
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
">
    <div style="font-size: 2rem;">🛒</div>
    <div>
        <div style="color: #fff; font-size: 1.3rem; font-weight: 700; letter-spacing: -0.02em;">
            Meridian Electronics
        </div>
        <div style="color: #a0aec0; font-size: 0.85rem; margin-top: 2px;">
            Customer Support — provide your email and 4-digit PIN to get started
        </div>
    </div>
</div>
"""

FOOTER_HTML = """
<div style="text-align: center; color: #718096; font-size: 0.75rem; padding: 0.75rem 0 0.25rem;">
    Meridian Electronics &nbsp;·&nbsp; Powered by GPT-4o-mini
</div>
"""

CUSTOM_CSS = """
/* Chat bubble area */
.chatbot .message-wrap { padding: 0.5rem 0.75rem; }

/* User bubbles */
.chatbot .message.user {
    background: #1a1a2e !important;
    color: #fff !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 0.65rem 1rem !important;
    max-width: 78% !important;
    margin-left: auto !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.15) !important;
}

/* Bot bubbles */
.chatbot .message.bot {
    background: #f0f4ff !important;
    color: #1a1a2e !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 0.65rem 1rem !important;
    max-width: 78% !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}

/* Input textarea */
.chatbot + * textarea, footer textarea {
    border-radius: 12px !important;
    border: 1.5px solid #cbd5e0 !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s !important;
    resize: none !important;
}
.chatbot + * textarea:focus, footer textarea:focus {
    border-color: #2b6cb0 !important;
    box-shadow: 0 0 0 3px rgba(43,108,176,0.12) !important;
    outline: none !important;
}

/* Send button */
#component-0 button.primary, .chat-interface button.primary {
    background: #1a1a2e !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: background 0.2s !important;
}
#component-0 button.primary:hover, .chat-interface button.primary:hover {
    background: #2b6cb0 !important;
}
"""


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
