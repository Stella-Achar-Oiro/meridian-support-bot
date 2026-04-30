"""
End-to-end test: drives all 4 flows against the live MCP server.
Run with:  python3 _e2e_test.py
"""
import asyncio
from dotenv import load_dotenv
load_dotenv()

from app import _chat_turn, SessionState

EMAIL = "donaldgarcia@example.net"
PIN   = "7912"

DIVIDER = "─" * 60

async def main() -> None:
    session = SessionState()

    async def turn(label: str, msg: str) -> str:
        print(f"\n{DIVIDER}")
        print(f"[USER]  {msg}")
        reply, _ = await _chat_turn(msg, session)
        print(f"[AGENT] {reply}")
        return reply

    print(f"\n{'═'*60}")
    print("FLOW 1 — Authentication")
    print(f"{'═'*60}")
    await turn("auth", f"Hi, my email is {EMAIL} and my PIN is {PIN}.")

    print(f"\n{'═'*60}")
    print("FLOW 2 — Browse products")
    print(f"{'═'*60}")
    await turn("list", "Can you show me all the products you carry?")

    print(f"\n{'═'*60}")
    print("FLOW 3 — Order history")
    print(f"{'═'*60}")
    await turn("orders", "What orders do I have on my account?")

    print(f"\n{'═'*60}")
    print("FLOW 4 — Place an order")
    print(f"{'═'*60}")
    await turn("order-start", "I'd like to order 1 unit of the first product you listed.")
    await turn("confirm",     "Yes, please go ahead and place the order.")

    print(f"\n{'═'*60}")
    print("All flows complete.")
    print(f"{'═'*60}")


if __name__ == "__main__":
    asyncio.run(main())
