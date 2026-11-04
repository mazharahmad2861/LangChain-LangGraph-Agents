
import os
import time

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


# =====================================================
# 1. ENVIRONMENT CONFIGURATION
# =====================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing from .env")


# =====================================================
# 2. DEFINE TOOLS
# =====================================================

@tool
def check_order_status(order_id: str) -> str:
    """Check the current status of a customer order."""

    orders = {
        "ORD101": "Shipped",
        "ORD102": "Processing",
        "ORD103": "Delivered",
    }

    return orders.get(
        order_id,
        "Order not found."
    )


@tool
def send_email(
    recipient: str,
    subject: str,
    body: str
) -> str:
    """Send an email to a customer."""

    # Mock implementation.
    # Replace with a real email service only after
    # configuring proper authentication and permissions.

    print("\n[EMAIL SERVICE]")
    print("Recipient:", recipient)
    print("Subject:", subject)
    print("Body:", body)

    return f"Email successfully sent to {recipient} (simulation)."


@tool
def get_customer_support_info(topic: str) -> str:
    """Retrieve customer support information."""

    information = {
        "refund": "Refunds generally take 5-7 business days.",
        "delivery": "Standard delivery takes 3-5 business days.",
        "cancellation": "Orders can be cancelled before dispatch."
    }

    return information.get(
        topic.lower(),
        "Please contact customer support for more information."
    )


# =====================================================
# 3. CUSTOM MIDDLEWARE: TOOL CALL LOGGING
# =====================================================

@wrap_tool_call
def logging_middleware(request, handler):
    """
    Log tool execution details and execution time.

    Avoid logging sensitive tool arguments.
    """

    tool_name = request.tool_call["name"]

    start_time = time.perf_counter()

    print("\n" + "=" * 50)
    print("[MIDDLEWARE] Tool execution started")
    print("Tool:", tool_name)

    try:
        result = handler(request)

        return result

    except Exception as error:
        print("[MIDDLEWARE] Tool failed:", type(error).__name__)
        raise

    finally:
        execution_time = time.perf_counter() - start_time

        print("Execution time:", round(execution_time, 3), "seconds")
        print("[MIDDLEWARE] Tool execution finished")
        print("=" * 50)


# =====================================================
# 4. MIDDLEWARE CONFIGURATION
# =====================================================

middleware_stack = [

    # MIDDLEWARE 1: PII PROTECTION
    PIIMiddleware(
        "credit_card",
        strategy="mask",
        apply_to_input=True,
        apply_to_tool_results=True
    ),

    # MIDDLEWARE 2: HUMAN APPROVAL
    HumanInTheLoopMiddleware(
        interrupt_on={
            "send_email": {
                "allowed_decisions": [
                    "approve",
                    "edit",
                    "reject"
                ]
            },

            # Read-only tools don't require approval
            "check_order_status": False,
            "get_customer_support_info": False
        }
    ),

    # MIDDLEWARE 3: TOOL CALL LIMIT
    ToolCallLimitMiddleware(
        run_limit=5,
        exit_behavior="continue"
    ),

    # MIDDLEWARE 4: CUSTOM LOGGING
    logging_middleware
]


# =====================================================
# 5. CREATE AGENT
# =====================================================

checkpointer = InMemorySaver()

agent = create_agent(

    model="openai:gpt-4.1-mini",

    tools=[
        check_order_status,
        send_email,
        get_customer_support_info
    ],

    system_prompt="""
    You are a professional customer support assistant.

    Your responsibilities:

    1. Help customers with order-related queries.
    2. Check order status using available tools.
    3. Provide information about refunds, delivery,
       and cancellations.
    4. Draft and send customer emails when requested.

    IMPORTANT RULES:

    - Always use tools when external information is needed.
    - Never invent order information.
    - Never claim an email was sent unless the tool
      successfully executes.
    - Be polite and concise.
    - Never expose sensitive customer information.
    - Sending emails requires human approval.
    """,

    middleware=middleware_stack,

    checkpointer=checkpointer
)


# =====================================================
# 6. CONFIGURE CONVERSATION
# =====================================================

config = {
    "configurable": {
        "thread_id": "customer_support_001"
    }
}


# =====================================================
# 7. RUN AGENT
# =====================================================

def run_agent(user_query: str):

    result = agent.invoke(

        {
            "messages": [
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        },

        config=config,
        version="v2"
    )

    # Check whether Human-in-the-Loop interrupted execution
    if result.interrupts:

        print("\n" + "=" * 50)
        print("HUMAN APPROVAL REQUIRED")
        print("=" * 50)

        for interrupt in result.interrupts:
            print(interrupt.value)

        decision = input(
            "\nApprove this action? (yes/no): "
        ).strip().lower()

        if decision == "yes":

            resume_command = Command(
                resume={
                    "decisions": [
                        {"type": "approve"}
                    ]
                }
            )

        else:

            resume_command = Command(
                resume={
                    "decisions": [
                        {
                            "type": "reject",
                            "message": (
                                "The human reviewer rejected "
                                "this email. Do not retry it."
                            )
                        }
                    ]
                }
            )

        result = agent.invoke(
            resume_command,
            config=config,
            version="v2"
        )

    print("\n" + "=" * 50)
    print("FINAL AGENT RESPONSE")
    print("=" * 50)

    # v2 GraphOutput stores the final state in .value
    messages = result.value["messages"]

    print(messages[-1].content)


# =====================================================
# 8. MAIN APPLICATION
# =====================================================

if __name__ == "__main__":

    print("Customer Support Agent Started!")
    print("Type 'exit' to quit.\n")

    while True:

        user_query = input("You: ").strip()

        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        if not user_query:
            continue

        try:
            run_agent(user_query)

        except Exception as error:
            print(
                "\nAgent execution failed:",
                type(error).__name__,
                str(error)
            )
