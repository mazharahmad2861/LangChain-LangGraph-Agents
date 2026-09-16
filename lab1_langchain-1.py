import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool


# ============================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================

load_dotenv()

# .env:
# OPENAI_API_KEY=your_api_key


# ============================================
# 2. CREATE MODEL
# ============================================

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)



# ============================================
# 3. NORMAL MODEL CALL - invoke()
# ============================================

response = llm.invoke(
    "In one sentence, what is an AI agent?"
)

print("NORMAL RESPONSE:")
print(response.content)


# ============================================
# 4. STREAMING
# ============================================

print("\nSTREAMING:")

for chunk in llm.stream(
    "Write a two-line poem about AI agents."
):
    print(chunk.content, end="", flush=True)

print()


# ============================================
# 5. MESSAGES
# ============================================

messages = [
    SystemMessage(
        "You are a helpful assistant."
    ),
    HumanMessage(
        "What is the capital of France?"
    )
]

response = llm.invoke(messages)

print("\nMESSAGE RESPONSE:")
print(response.content)


# ============================================
# 6. CREATE A TOOL
# ============================================

@tool
def get_share_price(symbol: str) -> float:
    """Return the current share price for a given ticker symbol."""

    fake_prices = {
        "AAPL": 241.5,
        "GOOG": 168.2,
        "AMZN": 198.0
    }

    return fake_prices.get(symbol.upper(),0.0)


# ============================================
# 7. CALL TOOL DIRECTLY
# ============================================

result = get_share_price.invoke(
    {"symbol": "AAPL"}
)

print("\nDIRECT TOOL CALL:")
print(result)


# ============================================
# 8. GIVE TOOL TO LLM
# ============================================

llm_with_tools = llm.bind_tools(
    [get_share_price]
)


# ============================================
# 9. MANUAL TOOL-CALLING LOOP
# ============================================

conversation = [
    HumanMessage(
        "What is the share price of Amazon?"
    )
]


# ---- First LLM call ----

ai_message = llm_with_tools.invoke(conversation)


conversation.append(ai_message)



# ---- Check tool requests ----

for call in ai_message.tool_calls:

    if call["name"] == "get_share_price":

        # Execute Python tool
        result = get_share_price.invoke(
            call["args"]
        )

        # Send result back to LLM
        conversation.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call["id"]
            )
        )


# ---- Second LLM call ----

final = llm_with_tools.invoke(
    conversation
)

print("\nFINAL TOOL RESPONSE:")
print(final.content)


# ============================================
# 10. STRUCTURED OUTPUT
# ============================================

class Company(BaseModel):

    name: str = Field(description="Company name")
    

    ticker: str = Field(description="Stock ticker symbol")
    

    founded_year: int = Field(description="Year company was founded")


structured_llm = llm.with_structured_output(Company)



company = structured_llm.invoke(
    "Tell me about Amazon the technology company."
)


print("\nSTRUCTURED OUTPUT:")

print(company)

print("Name:", company.name)
print("Ticker:", company.ticker)
print("Founded:", company.founded_year)