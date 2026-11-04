
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
from IPython.display import display, Image

load_dotenv()


# ==========================================
# 1. Create Tools
# ==========================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather information for a city."""

    weather_data = {
        "London": "Cloudy, 15°C",
        "Rome": "Sunny, 27°C"
    }

    return weather_data.get(city, "Weather information unavailable.")


@tool
def get_population(city: str) -> str:
    """Get the population of a city."""

    population_data = {
        "London": "8.9 million",
        "Rome": "2.8 million"
    }

    return population_data.get(
        city,
        "Population information unavailable."
    )


# ==========================================
# 2. Create Agent
# ==========================================

agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[
        get_weather,
        get_population
    ],
    system_prompt="""
    You are a helpful travel assistant.

    Use your available tools to answer questions
    about cities.

    Answer clearly and concisely.
    """
)


# ==========================================
# 3. Synchronous Execution
# ==========================================

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the weather and population of Rome?"
            }
        ]
    }
)

print("Agent Response:")
print(response["messages"][-1].content)


# ==========================================
# 4. Asynchronous Execution
# ==========================================

async def run_async():
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Compare the weather in London and Rome."
                }
            ]
        }
    )

    print("\nAsync Agent Response:")
    print(response["messages"][-1].content)


# Run in a normal Python script:
# import asyncio
# asyncio.run(run_async())

# In Jupyter Notebook:
# await run_async()


# ==========================================
# 5. Visualize Agent Graph
# ==========================================

graph = agent.get_graph()

print("\nAgent Graph:")
print(graph.draw_mermaid())

# Render graph in Jupyter Notebook:
display(Image(graph.draw_mermaid_png()))
