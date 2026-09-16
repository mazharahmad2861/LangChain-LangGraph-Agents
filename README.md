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

## Lab 2- LangGraph: Orchestration layer
