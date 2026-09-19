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

