# LangChain-LangGraph Agents and Workflows 
## Four abstraction levels 
Layer 1: Building blocks → manual control, langchain-core

Layer 2: LangGraph        → Orchestration layer (graph + state + memory), you control the flow

Layer 3: Agents: create_agent()   → automatic agent loop

Layer 4: Deep Agents: create_deep_agent()     → planning + sub-agents + filesystem  

## Lab 1- LangChain Layer  — Building Blocks 
### Core Concepts
#### Chat Models

Chat models generate responses to user messages.

Invoke: Returns a complete response.

Stream: Returns the response incrementally in chunks. 
#### Messages

System Message: Instructions for the model.

Human Message: User input.

AI Message: Model response, which may contain tool requests.

Tool Message: Result returned after a tool is executed. 
#### Manual Tool-Calling Loop

The basic agent process is:

1. Send the user message to the model.

2. Check whether the model requested any tools.

3. Execute the requested tools in Python.

4. Send the results back as tool messages.

5. Call the model again.

6. Repeat until the model returns a final answer without requesting more tools.

Multiple tools can be bound to the same model, and the model may request several tools in one response. 
#### Structured Output

Structured output makes the model return data in a predefined schema, such as a Pydantic model with specific fields and types.

It is useful for:

Data extraction,
Classification,
API responses,
Database operations,
Automation,
Agent state

#### Output Parser vs Structured Output

Output Parser: Converts or parses an already-generated model response.

Structured Output: Requests the model to produce data in a defined structure directly.  

### Main Takeaway

Layer 1 provides the fundamental components used by higher-level agent frameworks:

##### Models + Messages + Tools + Tool-Calling Loop + Structured Output 

# Lab 2
# LangGraph — Orchestration Layer

## Overview

LangGraph is used to orchestrate **stateful, multi-step AI workflows** using nodes, edges, tools, and checkpoints.

--- 
## Architecture

```text
     START
       ↓
     Chatbot
       ↓
    Tool Call?
 ┌───────────────┐
 │               │
Yes             No
 ↓               ↓
Tools        Translator
 ↓               ↓
Chatbot          END
 ↓
Tool Call?
 ├── Yes → Tools
 └── No  → END
```

The chatbot can repeatedly call tools until no further tool is required, after which the workflow terminates.


## State & Reducers

### State

Shared data that flows between nodes.

### `Annotated`

Used to attach a reducer/instruction to a state field.

`Annotated[list, add_messages]` means:

* `list` → type of the field
* `add_messages` → how updates should be merged

### Reducer

Defines how new state updates are combined with existing state instead of simply replacing it.

---

## Tool Calling

LangGraph supports the **LLM → Tool → LLM** pattern.

### `ToolNode`

Executes tools requested by the LLM.

### `tools_condition`

Routes execution based on whether the LLM requested a tool.

Typical flow:

**Chatbot → Tools → Chatbot → END**

This loop can repeat when multiple tools are required.

---

## Conditional Routing

Conditional edges allow the graph to choose the next node dynamically.

Example:

**Chatbot → Tools** if a tool is requested

**Chatbot → Translator** if no tool is required

This is the foundation for branching workflows.

---

## Multiple Nodes

Different responsibilities can be separated into different nodes.

Example:

**Chatbot → Tools → Chatbot → Translator → END**

Each node handles a specific part of the workflow.

---

## Memory & Checkpointing

A normal graph invocation starts with fresh state unless checkpointing is configured.

### Checkpointer

Saves graph state at different stages of execution.

### `MemorySaver`

Stores checkpoints in memory.

### `SqliteSaver`

Stores checkpoints in SQLite, allowing state to persist across application restarts.

### Thread ID

Identifies a conversation/workflow.

**Thread ID = conversation**

**Checkpoint = saved state within that conversation**

---

## State Inspection

### `get_state()`

Retrieves the latest saved state.

> **Where am I now?**

### `get_state_history()`

Retrieves previous checkpoints.

> **Where have I been?**

A checkpoint contains the state at a particular point in graph execution.

---

## Time Travel

LangGraph can resume execution from an earlier checkpoint.

Process:

**History → Select checkpoint → Get checkpoint ID → Resume execution**

Time travel can also create a **new branch** from an earlier state.

Example:

**A → B → C → D**

Go back to **B** and continue differently:

**A → B → X → Y**

The original execution is not changed.

### Mental Model

**Checkpoint = Save point**

**Checkpoint ID = Version**

**Replay = Resume from a previous save point**

**Branch = Resume with different input**

---

## LangSmith

Used for **observability and tracing**.

Useful for inspecting:

* LLM calls
* Tool calls
* Graph execution
* Inputs and outputs

---

## Key Takeaways

* **State** → shared workflow data
* **Reducer** → controls state updates
* **Node** → unit of work
* **Conditional edge** → dynamic routing
* **ToolNode** → tool execution
* **tools_condition** → tool-based routing
* **Checkpointer** → saves state
* **Thread ID** → identifies a conversation
* **Checkpoint ID** → identifies a saved state
* **`get_state()`** → current state
* **`get_state_history()`** → previous states
* **Time travel** → resume from an earlier checkpoint
* **LangSmith** → tracing and observability

## Mental Model

**LangGraph = State + Nodes + Edges + Tools + Checkpoints**

It provides the orchestration layer required to build **stateful and controllable agent workflows**. 

# Lab 3 LangChain Agents & Browser Automation with Playwright MCP

## Overview

Day 3 focuses on building AI agents using LangChain's `create_agent()` and extending their capabilities through middleware, memory, structured output, and external tools.

The practical implementation involves building a **Web Research Agent** using Playwright MCP, enabling an LLM to interact with websites through browser automation.

## 1. LangChain `create_agent()`

High-level API for creating tool-calling agents.

**Key components:**
- Model – LLM responsible for reasoning and decision-making.
- Tools – External functions available to the agent.
- System Prompt – Defines the agent's role and behavior.
- Middleware – Customizes and controls agent execution.
- Checkpointer – Saves and restores agent state.
- Response Format – Defines structured output.

**Execution methods:**
- `invoke()` – Synchronous execution.
- `ainvoke()` – Asynchronous execution.

`create_agent()` uses LangGraph internally for agent execution and state management.

## 2. Memory & Checkpointing

LangGraph checkpointing allows agents to preserve conversational state across interactions.

| Concept | Purpose |
|---|---|
| Checkpointer | Saves and restores agent state. |
| Thread ID | Identifies a conversation. |
| InMemorySaver | Stores checkpoints in memory; useful for development. |
| Persistent Checkpointer | Stores state in a database for long-term persistence. |

**Important:** Reusing the same thread ID allows the agent to continue a previous conversation.

## 3. Structured Output

Structured output ensures that an agent returns data in a predefined format instead of unrestricted text.

- Implemented using Pydantic models.
- Defines fields, data types, and descriptions.
- Supports output validation.
- Useful for API responses, database storage, and downstream processing.

## 4. Middleware

Middleware allows developers to intercept and customize different stages of an agent's execution.

It acts as an additional control layer around model calls and tool execution, enabling logging, security, privacy protection, human approval, and resource management.

### Types of Middleware

| Middleware | Purpose | Example Use Case |
|---|---|---|
| Custom Middleware | Implements custom logic around agent execution. | Logging, input validation, monitoring tool calls. |
| Logging Middleware | Records tool names, arguments, results, and execution details. | Debugging and tracking agent behavior. |
| PIIMiddleware | Detects sensitive information and applies protection strategies. | Masking credit card numbers, emails, and phone numbers. |
| HumanInTheLoopMiddleware | Pauses selected tool operations until human approval is received. | Approving emails before sending or confirming file deletion. |
| ToolCallLimitMiddleware | Restricts the number of tool executions. | Preventing excessive tool calls and infinite execution loops. |
| SummarizationMiddleware | Summarizes older conversation messages when configured context thresholds are reached. | Managing long conversations and reducing token usage. |

### Important Concepts

**1. Custom Middleware (`@wrap_tool_call`)**
- Wraps tool execution with custom logic.
- Provides access to tool names and arguments.
- Can log execution details, validate requests, and handle errors.
- Calls the actual tool handler to continue execution.

**2. PII Middleware**

PII (Personally Identifiable Information) middleware detects sensitive information and applies configured protection strategies.

Common strategies:
- `mask` – Replaces sensitive characters with masking characters.
- `redact` – Removes sensitive information.
- `block` – Prevents processing when sensitive information is detected.

Custom regex detectors can be configured for Indian-specific information such as Aadhaar, PAN, phone numbers, and bank account numbers.

**3. Human-in-the-Loop (HITL)**

Allows an agent to pause before executing selected tools and request human intervention.

Possible decisions:
- Approve – Execute the tool.
- Reject – Cancel the tool execution.
- Edit – Modify the proposed tool arguments before execution, where supported.

Requires checkpointing to preserve the interrupted agent state and resume execution.

**4. Tool Call Limits**

Controls how many times an agent can execute tools, helping prevent unnecessary tool usage, excessive API calls, and uncontrolled execution loops.

**5. Summarization Middleware**

Automatically summarizes older conversation history based on configured token thresholds, helping manage context-window limitations and long-running conversations.

### Why Middleware Matters

- Adds security and privacy controls.
- Provides visibility into agent execution.
- Prevents uncontrolled tool usage.
- Enables human approval for sensitive operations.
- Makes agent behavior customizable and manageable.

**Key takeaway:** Middleware provides a way to control, monitor, secure, and customize agent execution without rewriting the entire agent logic.

## 5. Node.js & Playwright

**Node.js:** JavaScript runtime required to execute the Playwright MCP server.

Important commands:
- `node` – Executes JavaScript.
- `npm` – Manages Node.js packages.
- `npx` – Executes packages without global installation.

**Playwright:** Microsoft's browser automation framework.

Capabilities:
- Browser navigation.
- Clicking buttons and links.
- Typing into input fields.
- Keyboard interactions.
- Reading webpage content.
- Taking screenshots.

## 6. Model Context Protocol (MCP)

MCP (Model Context Protocol) standardizes communication between AI applications and external tools.

**Core components:**
- MCP Host – Application providing the environment for MCP.
- MCP Client – Connects to MCP servers.
- MCP Server – Exposes tools and capabilities.
- MCP Tools – Operations available to the AI application.

**STDIO Transport:**
- Communication through standard input and output.
- Client launches the local MCP server.
- Commonly used for local MCP integrations.

## 7. Playwright MCP Integration

Playwright MCP exposes browser automation capabilities as tools that an AI agent can use.

**Important components:**
- `MultiServerMCPClient` – Connects to MCP servers.
- `get_tools()` – Retrieves available MCP tools.
- `client.session()` – Establishes a managed MCP session.
- `load_mcp_tools()` – Loads tools from an active session.

**Important:** Maintain an active MCP session while the agent executes browser operations to preserve browser context.

### Common Browser Tools

| Tool | Function |
|---|---|
| `browser_navigate` | Opens a URL. |
| `browser_click` | Clicks an element. |
| `browser_type` | Types text. |
| `browser_snapshot` | Retrieves the accessibility snapshot of a webpage. |
| `browser_take_screenshot` | Captures a screenshot. |
| `browser_press_key` | Performs keyboard actions. |
| `browser_navigate_back` | Navigates to the previous page. |

## 8. Web Research Agent

**Project:** Build a Web Research Agent using LangChain `create_agent()` and Playwright MCP.

**Capabilities:**
- Visit websites and navigate pages.
- Extract information from webpages.
- Perform browser interactions.
- Research topics using live website content.
- Summarize and present collected information.

**Important considerations:**
- Website content is untrusted input.
- Avoid blindly following instructions embedded in webpages.
- Require authorization for sensitive browser actions.
- Use LangSmith for tracing and debugging agent execution.

## 9. Key Takeaways

- `create_agent()` simplifies tool-calling agent development.
- LangGraph handles underlying agent execution and state management.
- Checkpointers enable conversation continuity.
- Pydantic enables structured and validated responses.
- Middleware adds security, monitoring, and execution controls.
- Node.js runs the Playwright MCP server.
- Playwright automates browser interactions.
- MCP connects AI applications with external tools.
- Playwright MCP makes browser tools available to LangChain agents.
- LangSmith helps debug and monitor agent execution.

