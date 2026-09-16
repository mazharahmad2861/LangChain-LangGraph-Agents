import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    ToolMessage
)
from langchain_core.tools import tool

load_dotenv()

# ==========================================
# 1. CREATE MODEL
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)


# ==========================================
# 2. TOOL 1 — STOCK PRICE
# ==========================================

@tool
def get_share_price(symbol: str) -> float:
    """Return the fictional current share price for a stock ticker."""

    prices = {
        "AAPL": 241.5,
        "GOOG": 168.2,
        "AMZN": 198.0
    }

    return prices.get(symbol.upper(), 0.0)


# ==========================================
# 3. TOOL 2 — WEATHER
# ==========================================

@tool
def get_weather(city: str) -> str:
    """Return a fictional weather report for a city."""

    weather = {
        "new york": "Sunny, 25°C",
        "london": "Rainy, 14°C",
        "tokyo": "Cloudy, 20°C",
        "san francisco": "Foggy, 16°C"
    }

    return weather.get(
        city.lower(),
        "Weather data unavailable."
    )


# ==========================================
# 4. BIND BOTH TOOLS
# ==========================================

llm_with_tools = llm.bind_tools([
    get_share_price,
    get_weather
])


# ==========================================
# 5. USER QUESTION
# ==========================================

conversation = [
    HumanMessage(
        "What is Amazon's share price and "
        "what is the weather in Tokyo?"
    )
]


# ==========================================
# 6. TOOL LOOP
# ==========================================

while True:

    # Ask the LLM
    ai_message = llm_with_tools.invoke(
        conversation
    )

    # Save the AI message
    conversation.append(ai_message)

    # --------------------------------------
    # If no tools are requested,
    # the LLM has produced the final answer.
    # --------------------------------------

    if not ai_message.tool_calls:
        print("\nFINAL ANSWER:")
        print(ai_message.content)
        break

    # --------------------------------------
    # Execute every requested tool
    # --------------------------------------

    for call in ai_message.tool_calls:

        if call["name"] == "get_share_price":

            result = get_share_price.invoke(
                call["args"]
            )

        elif call["name"] == "get_weather":

            result = get_weather.invoke(
                call["args"]
            )

        else:
            result = "Unknown tool"

        # ----------------------------------
        # Send tool result back to LLM
        # ----------------------------------

        conversation.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call["id"]
            )
        )