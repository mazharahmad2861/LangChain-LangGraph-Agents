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


# Lab 4 Deep Agents 

## Overview

**Deep Agents** is a higher-level agent framework built on top of LangChain and LangGraph for developing agents capable of handling complex, multi-step tasks.

Unlike basic tool-calling agents, Deep Agents provides a built-in agent harness that supports planning, task delegation, filesystem management, reusable skills, and context management.

It is designed for tasks that require more than a single interaction, such as research, report generation, analysis, and content creation.

## 1. Understanding the LangChain Ecosystem

The ecosystem can be understood through four layers:

- **LangChain Core:** Provides the fundamental building blocks, including models, messages, prompts, and tools.
- **LangGraph:** Handles agent orchestration, state management, and workflow execution.
- **LangChain `create_agent`:** Simplifies the creation of tool-calling agents.
- **Deep Agents:** Provides a more comprehensive agent harness for managing complex tasks through planning, delegation, and additional built-in capabilities.

## 2. Core Capabilities of Deep Agents

| Capability | Purpose |
|---|---|
| Planning | Breaks complex tasks into smaller steps using a to-do list. |
| Filesystem | Reads, writes, edits, and organizes files. |
| Sub-agents | Delegates specialized tasks to other agents. |
| Skills | Provides reusable instructions and domain-specific knowledge. |
| Tools | Enables interaction with external systems and custom functionality. |
| Middleware | Adds additional behavior and control to agent execution. |

## 3. How Deep Agents Work

A typical Deep Agent follows this process:

1. **Understand the task:** Interprets the user's objective.
2. **Plan:** Breaks the objective into manageable subtasks.
3. **Execute:** Uses available tools and filesystem operations.
4. **Delegate:** Assigns specialized tasks to sub-agents when required.
5. **Combine results:** Integrates the outputs from different tasks.
6. **Generate output:** Produces the final report, presentation, or other requested artifact.

The agent can revisit its plan and use intermediate files to manage longer workflows.

## 4. Important Concepts

### Filesystem Backend

The filesystem backend provides the agent with a workspace to store and manage information.

It supports operations such as reading, writing, editing, and listing files. A sandbox can be used to restrict file operations to a designated directory.

With `virtual_mode=True`, the configured directory is exposed as the agent's virtual root (`/`).

### Skills

Skills are reusable instruction packages that teach an agent how to perform a particular task.

A skill typically contains a `SKILL.md` file with metadata and detailed instructions. Additional resources can also be included.

**Progressive disclosure** allows the agent to load detailed instructions only when they are relevant, rather than loading every skill upfront.

### Sub-agents

Sub-agents are specialized agents created to handle specific parts of a larger task.

A lead agent can delegate work to sub-agents using the task capability. Each sub-agent operates with its own execution context and returns results to the lead agent.

This enables modular development and parallel execution of independent tasks.

### Custom Tools

Custom tools connect an agent's reasoning capabilities to actual Python functionality or external systems.

A Python function can be exposed as a tool, allowing the agent to invoke operations such as generating PowerPoint presentations, processing data, or interacting with APIs.

The agent decides when to use the tool, while the underlying function performs the actual operation.

### Middleware

Middleware provides additional control over agent execution and can introduce behavior such as summarization, human approval, and other execution-level controls.

### LangSmith Tracing

LangSmith helps inspect agent execution and understand how an agent interacts with tools, middleware, and sub-agents.

It is particularly useful for debugging complex workflows and understanding nested delegation.

## 5. Understanding the Architecture

A typical Deep Agents application can be organized into these components:

- **Main Agent:** Understands the overall objective and coordinates execution.
- **Sub-agents:** Handle specialized responsibilities.
- **Skills:** Provide task-specific instructions.
- **Tools:** Perform concrete operations.
- **Filesystem Backend:** Stores intermediate and final outputs.
- **Middleware:** Adds execution controls and additional behavior.

These components work together to build agents capable of completing complex tasks.

## 6. Practical Applications

The concepts covered in this project can be applied to:

- **AI Research Assistant:** Researches topics and generates structured reports.
- **Multi-Agent Competitor Analysis:** Delegates research on different competitors and combines the findings.
- **Automated Presentation Generator:** Uses research reports, presentation skills, and custom tools to generate PowerPoint slides.

## Key Takeaways

- Deep Agents provides a structured environment for building complex AI agents.
- Planning helps organize multi-step tasks.
- Skills make agent instructions reusable and modular.
- Sub-agents enable task delegation and specialization.
- Custom tools connect agent reasoning with real-world operations.
- Filesystem backends help manage intermediate and final outputs.
- Middleware and LangSmith improve control, observability, and debugging.

**The main idea:** Deep Agents moves beyond simple question-answering and tool calling toward building structured, extensible agents that can plan, delegate, execute, and produce meaningful outputs.
