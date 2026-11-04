
# LANGGRAPH

import os
import requests

from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper

from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

import gradio as gr


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(override=True)

# Required environment variables from the chapter:
#
# OPENAI_API_KEY
# SERPER_API_KEY
# PUSHOVER_TOKEN
# PUSHOVER_USER
#
# Optional LangSmith variables:
#
# LANGSMITH_TRACING=true
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# LANGSMITH_API_KEY=...
# LANGSMITH_PROJECT=agentic-track


# ============================================================
# 3. DEFINE STATE
# ============================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    spanish: str


# ============================================================
# 4. CREATE LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-5.4-mini"
)


# ============================================================
# 5. CREATE TOOLS
# ============================================================

# ------------------------------------------------------------
# 5A. PREBUILT SEARCH TOOL
# ------------------------------------------------------------

search = GoogleSerperRun(
    api_wrapper=GoogleSerperAPIWrapper()
)


# ------------------------------------------------------------
# 5B. CUSTOM PUSH NOTIFICATION TOOL
# ------------------------------------------------------------

@tool
def send_push_notification(text: str) -> str:
    """Send a short push notification to the user's phone."""

    response = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        },
    )

    return "Notification sent"


# ============================================================
# 6. PUT ALL TOOLS INTO A LIST
# ============================================================

tools = [
    search,
    send_push_notification,
]


# ============================================================
# 7. BIND TOOLS TO THE LLM
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 8. CHATBOT NODE
# ============================================================

def chatbot_node(state: State) -> dict:

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# ============================================================
# 9. TRANSLATOR NODE
# ============================================================

def translator_node(state: State) -> dict:

    # Get the final message from the conversation
    last_message = state["messages"][-1].content

    # Build translation prompt
    prompt = (
        "Translate this into Spanish, "
        "replying with the translation only:\n\n"
        f"{last_message}"
    )

    # Call LLM
    translation = llm.invoke(prompt)

    return {
        "spanish": translation.content
    }


# ============================================================
# 10. BUILD THE LANGGRAPH
# ============================================================

builder = StateGraph(State)


# ------------------------------------------------------------
# Add nodes
# ------------------------------------------------------------

builder.add_node(
    "chatbot",
    chatbot_node
)

builder.add_node(
    "tools",
    ToolNode(tools)
)

builder.add_node(
    "translator",
    translator_node
)


# ------------------------------------------------------------
# START → CHATBOT
# ------------------------------------------------------------

builder.add_edge(
    START,
    "chatbot"
)


# ------------------------------------------------------------
# CHATBOT → TOOLS OR TRANSLATOR
# ------------------------------------------------------------

builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        "tools": "tools",
        END: "translator",
    }
)


# ------------------------------------------------------------
# TOOLS → CHATBOT
# ------------------------------------------------------------

builder.add_edge(
    "tools",
    "chatbot"
)


# ------------------------------------------------------------
# TRANSLATOR → END
# ------------------------------------------------------------

builder.add_edge(
    "translator",
    END
)


# ============================================================
# 11. ADD MEMORY / CHECKPOINTING
# ============================================================

memory = MemorySaver()


graph = builder.compile(
    checkpointer=memory
)


# ============================================================
# 12. BASIC GRAPH INVOCATION
# ============================================================

config = {
    "configurable": {
        "thread_id": "conversation-1"
    }
}


result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "In one sentence, what is special about a banana?"
            }
        ]
    },
    config
)


# ------------------------------------------------------------
# Print final answer
# ------------------------------------------------------------

print("English:")
print(result["messages"][-1].content)

print()

print("Spanish:")
print(result["spanish"])


# ============================================================
# 13. TEST TOOL-USING AGENT
# ============================================================

tool_result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use your search tool to tell me "
                    "the ingredients in biryani "
                    "and send me a push notification."
                ),
            }
        ]
    },
    {
        "configurable": {
            "thread_id": "conversation-tools"
        }
    }
)


print("\nTool-based result:")
print(
    tool_result["messages"][-1].content
)

print("\nSpanish:")
print(
    tool_result["spanish"]
)


# ============================================================
# 14. TEST MEMORY
# ============================================================

memory_config = {
    "configurable": {
        "thread_id": "memory-demo"
    }
}


# First conversation
graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hi, my name is Mazhar."
            }
        ]
    },
    memory_config
)


# Second conversation
memory_result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is my name?"
            }
        ]
    },
    memory_config
)


print("\nMemory result:")
print(
    memory_result["messages"][-1].content
)

print("\nSpanish:")
print(
    memory_result["spanish"]
)


# ============================================================
# 15. INSPECT CURRENT STATE
# ============================================================

snapshot = graph.get_state(
    memory_config
)


print("\nCurrent state:")
print(snapshot.values)


# ============================================================
# 16. INSPECT MESSAGES
# ============================================================

messages = snapshot.values["messages"]


print("\nMessages stored:")
print(
    "Number of messages:",
    len(messages)
)


for message in messages:

    print(
        type(message).__name__,
        ":",
        message.content
    )


# ============================================================
# 17. GET CHECKPOINT HISTORY
# ============================================================

history = list(
    graph.get_state_history(
        memory_config
    )
)


print("\nNumber of checkpoints:")
print(len(history))


# ============================================================
# 18. INSPECT CHECKPOINTS
# ============================================================

for h in reversed(history):

    metadata = h.metadata

    queued = ", ".join(
        task.name
        for task in h.tasks
    ) or "(run complete)"

    print(
        f"step {metadata.get('step'):>2} "
        f"source={metadata.get('source')} "
        f"messages={len(h.values.get('messages', []))} "
        f"about to run: {queued}"
    )


# ============================================================
# 19. TIME TRAVEL
# ============================================================

if history:

    # Pick an earlier checkpoint
    earlier = history[
        len(history) // 2
    ]

    print(
        "\nEarlier checkpoint contained",
        len(
            earlier.values["messages"]
        ),
        "messages"
    )


    # Extract checkpoint ID
    checkpoint_id = (
        earlier
        .config["configurable"]
        ["checkpoint_id"]
    )


    replay_config = {
        "configurable": {
            "thread_id": "memory-demo",
            "checkpoint_id": checkpoint_id,
        }
    }


    # Resume from that checkpoint
    resumed = graph.invoke(
        None,
        replay_config
    )


    print(
        "\nResumed graph contains",
        len(
            resumed["messages"]
        ),
        "messages"
    )


# ============================================================
# 20. PERSISTENT SQLITE MEMORY
# ============================================================

with SqliteSaver.from_conn_string(
    "memory.db"
) as sql_memory:

    durable_graph = builder.compile(
        checkpointer=sql_memory
    )


    sql_config = {
        "configurable": {
            "thread_id": "conversation-sqlite"
        }
    }


    # First invocation
    durable_graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Remember that my "
                        "favorite color is orange."
                    ),
                }
            ]
        },
        sql_config
    )


    # Second invocation
    reply = durable_graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What is my favorite color?"
                    ),
                }
            ]
        },
        sql_config
    )


    print("\nSQLite memory result:")

    print(
        reply["messages"][-1].content
    )

    print(
        "\nSpanish:"
    )

    print(
        reply["spanish"]
    )


# ============================================================
# 21. GRADIO UI
# ============================================================

def chat(
    user_input: str,
    history
):

    config = {
        "configurable": {
            "thread_id": "gradio-session4"
        }
    }


    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        },
        config
    )


    english_answer = (
        result["messages"][-1].content
    )

    spanish_answer = (
        result["spanish"]
    )


    return (
        f"{english_answer}\n\n"
        f"*{spanish_answer}*"
    )


# ============================================================
# 22. LAUNCH GRADIO
# ============================================================

gr.ChatInterface(chat).launch() 