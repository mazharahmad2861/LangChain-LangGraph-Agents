
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

from langchain.agents.middleware import (
    PIIMiddleware,
    HumanInTheLoopMiddleware,
    ToolCallLimitMiddleware,
    wrap_tool_call,
)

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()


# ==========================================
# 1. TOOLS
# ==========================================

@tool
def check_order(order_id: str) -> str:
    """Check the status of a customer order."""

    orders = {
        "ORD101": "Shipped",
        "ORD102": "Processing",
        "ORD103": "Delivered"
    }

    return orders.get(order_id, "Order not found")


@tool
def send_email(email: str, message: str) -> str:
    """Send an email to a customer."""

    print(f"\nSending email to: {email}")
    print(f"Message: {message}")

    return "Email sent successfully (simulation)."


@tool
def get_refund_policy() -> str:
    """Get the company's refund policy."""

    return "Refunds are processed within 5-7 business days."


# ==========================================
# 2. CUSTOM LOGGING MIDDLEWARE
# ==========================================

@wrap_tool_call
def logging_middleware(request, handler):

    tool_name = request.tool_call["name"]

    print(f"\n[LOG] Calling tool: {tool_name}")
    print(f"[LOG] Arguments: {request.tool_call['args']}")

    result = handler(request)

    print(f"[LOG] Tool completed: {tool_name}")

    return result


# ==========================================
# 3. CHECKPOINTER
# ==========================================

checkpointer = InMemorySaver()


# ==========================================
# 4. CREATE AGENT WITH MIDDLEWARE
# ==========================================

agent = create_agent(

    model="openai:gpt-4.1-mini",

    tools=[
        check_order,
        send_email,
        get_refund_policy
    ],

    system_prompt="""
    You are a helpful customer support assistant.

    - Use tools to answer customer questions.
    - Never invent order information.
    - Ask for approval before sending emails.
    - Provide clear and concise answers.
    """,

    middleware=[

        # 1. Mask sensitive credit card information
        PIIMiddleware(
            "credit_card",
            strategy="mask"
        ),

        # 2. Require human approval before sending emails
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True
            }
        ),

        # 3. Limit tool calls
        ToolCallLimitMiddleware(
            run_limit=5
        ),

        # 4. Log tool execution
        logging_middleware
    ],

    checkpointer=checkpointer
)


# ==========================================
# 5. RUN AGENT
# ==========================================

config = {
    "configurable": {
        "thread_id": "customer_001"
    }
}


def chat(user_query):

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        },
        config=config
    )

    # Check for human approval interrupt
    if result.get("__interrupt__"):

        print("\n[HUMAN APPROVAL REQUIRED]")

        for interrupt in result["__interrupt__"]:
            print(interrupt.value)

        decision = input(
            "\nApprove action? (yes/no): "
        ).strip().lower()

        if decision == "yes":
            decision_data = {"type": "approve"}
        else:
            decision_data = {
                "type": "reject",
                "message": "Action rejected by human."
            }

        # Resume agent execution
        result = agent.invoke(
            Command(
                resume={
                    "decisions": [decision_data]
                }
            ),
            config=config
        )

    print("\nAgent:", result["messages"][-1].content)


# ==========================================
# 6. CHAT LOOP
# ==========================================

if __name__ == "__main__":

    print("Customer Support Agent")
    print("Type 'exit' to stop.\n")

    while True:

        query = input("You: ")

        if query.lower() == "exit":
            break

        try:
            chat(query)

        except Exception as e:
            print("Error:", e)
