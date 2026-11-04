
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call

from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


# ==========================================
# 1. Define Tools
# ==========================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""

    weather = {
        "London": "Rainy, 14°C",
        "Rome": "Sunny, 27°C"
    }

    return weather.get(city, "Weather unavailable")


@tool
def get_population(city: str) -> str:
    """Get the population of a city."""

    population = {
        "London": "8.9 million",
        "Rome": "2.8 million"
    }

    return population.get(city, "Population unavailable")


# ==========================================
# 2. Define Structured Output
# ==========================================

class CityReport(BaseModel):
    city: str = Field(description="City name")

    weather: str = Field(
        description="Current weather description"
    )

    population: str = Field(
        description="City population"
    )


# ==========================================
# 3. Define Middleware
# ==========================================

@wrap_tool_call
def logging_middleware(request, handler):

    print(
        f"\nMiddleware: Calling {request.tool_call['name']}"
    )

    print(
        f"Arguments: {request.tool_call['args']}"
    )

    result = handler(request)

    print("Tool execution completed")

    return result


# ==========================================
# 4. Create Checkpointer
# ==========================================

checkpointer = InMemorySaver()


# ==========================================
# 5. Create Agent
# ==========================================

agent = create_agent(
    model="openai:gpt-4.1-mini",

    tools=[
        get_weather,
        get_population
    ],

    system_prompt="""
    You are a helpful travel assistant.

    Use your tools to retrieve city information.
    Remember previous conversation context.
    Answer accurately.
    """,

    response_format=CityReport,

    middleware=[
        logging_middleware
    ],

    checkpointer=checkpointer
)


# ==========================================
# 6. Configure Conversation
# ==========================================

config = {
    "configurable": {
        "thread_id": "travel_session_001"
    }
}


# ==========================================
# 7. First Interaction
# ==========================================

response1 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "I'm planning a trip to London."
            }
        ]
    },
    config=config
)

print("\nFirst Response:")
print(response1["messages"][-1].content)


# ==========================================
# 8. Second Interaction
# ==========================================

response2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like where I'm going?"
            }
        ]
    },
    config=config
)

print("\nSecond Response:")
print(response2["messages"][-1].content)


# ==========================================
# 9. Structured Output
# ==========================================

report_agent = create_agent(
    model="openai:gpt-4.1-mini",

    tools=[
        get_weather,
        get_population
    ],

    system_prompt="""
    You are a city information assistant.
    Use both tools to prepare the city report.
    """,

    response_format=CityReport
)


result = report_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Give me a report on London."
            }
        ]
    }
)

report = result["structured_response"]

print("\nStructured Report:")
print(report)

print("\nIndividual Fields:")
print("City:", report.city)
print("Weather:", report.weather)
print("Population:", report.population)

print("\nObject Type:")
print(type(report))
