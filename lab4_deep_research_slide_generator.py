
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from slide_kit import build_slide


# =====================================================
# 1. ENVIRONMENT CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

SANDBOX_DIR = PROJECT_DIR / "sandbox"

SKILLS_DIR = SANDBOX_DIR / "skills"

SANDBOX_DIR.mkdir(exist_ok=True)
SKILLS_DIR.mkdir(exist_ok=True)

load_dotenv(PROJECT_DIR / ".env", override=True)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing")

if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY is missing")


# =====================================================
# 2. MODEL
# =====================================================

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
    temperature=0
)


# =====================================================
# 3. SEARCH TOOL
# =====================================================

search = GoogleSerperRun(
    api_wrapper=GoogleSerperAPIWrapper()
)


# =====================================================
# 4. FILESYSTEM BACKEND
# =====================================================

backend = FilesystemBackend(
    root_dir=str(SANDBOX_DIR),
    virtual_mode=True
)


# =====================================================
# 5. RESEARCH AGENT
# =====================================================

research_prompt = """
You are an expert research analyst.

Your responsibility is to research the user's topic
and prepare a structured Markdown report.

Instructions:

1. Understand the research objective.
2. Plan the task using your todo tool.
3. Break the topic into smaller questions.
4. Search the internet using the search tool.
5. Collect relevant information from reliable sources.
6. Compare findings and identify important insights.
7. Do not invent statistics or facts.
8. Include sources wherever possible.
9. Write a clear and structured Markdown report.

Save the final report as:

/research_report.md
"""


research_agent = create_deep_agent(

    model=model,

    tools=[search],

    system_prompt=research_prompt,

    backend=backend
)


# =====================================================
# 6. CUSTOM POWERPOINT TOOL
# =====================================================

@tool
def create_presentation(
    title: str,
    key_points: list[str],
    recommendation: str
) -> str:
    """
    Create a professional PowerPoint presentation
    using the supplied title, key points and takeaway.
    """

    output_path = SANDBOX_DIR / "research_presentation.pptx"

    build_slide(
        title=title,
        key_points=key_points,
        recommendation=recommendation,
        output_path=str(output_path)
    )

    return (
        "Presentation successfully created at "
        "/research_presentation.pptx"
    )


# =====================================================
# 7. SLIDE-MAKER SUB-AGENT
# =====================================================

slide_maker_prompt = """
You are a professional business presentation designer.

Your task is to convert the research report into
a concise and professional PowerPoint presentation.

Instructions:

1. Read /research_report.md using filesystem tools.
2. Find and retrieve the relevant presentation skill.
3. Follow the presentation skill carefully.
4. Extract the report title.
5. Select up to five important findings.
6. Identify the main takeaway or recommendation.
7. Call the create_presentation tool.
8. Confirm that the presentation was generated.

Do not invent findings.

Use only information supported by the report.
"""


slide_maker = {

    "name": "slide-maker",

    "description": (
        "Reads a research report and converts "
        "it into a professional PowerPoint presentation."
    ),

    "system_prompt": slide_maker_prompt,

    "tools": [create_presentation],

    "skills": ["/skills/"]
}


# =====================================================
# 8. PRESENTATION LEAD AGENT
# =====================================================

presentation_prompt = """
You are a presentation workflow coordinator.

You have access to a specialized slide-maker sub-agent.

Your responsibility is to:

1. Understand the presentation request.
2. Delegate the presentation task to slide-maker.
3. Ensure the sub-agent reads the research report.
4. Ensure it follows the presentation skill.
5. Receive the final result.

Do not create the PowerPoint yourself.
Delegate this task to slide-maker.
"""


presentation_agent = create_deep_agent(

    model=model,

    subagents=[slide_maker],

    system_prompt=presentation_prompt,

    backend=backend
)


# =====================================================
# 9. HELPER: INSPECT TOOL CALLS
# =====================================================

def inspect_tool_calls(result):

    tools_used = []

    for message in result.get("messages", []):

        calls = getattr(message, "tool_calls", []) or []

        for call in calls:
            tools_used.append(call["name"])

    print("\nTools called by this agent:")

    for tool_name in tools_used:
        print("-", tool_name)


# =====================================================
# 10. RESEARCH WORKFLOW
# =====================================================

def run_research(topic):

    research_task = f"""
    Research the following topic:

    {topic}

    Prepare a structured report containing:

    1. Introduction
    2. Current Landscape
    3. Major Developments
    4. Applications
    5. Challenges
    6. Opportunities
    7. Future Trends
    8. Conclusion
    9. Sources

    Use your todo tool to plan the work.

    Search the internet for relevant information.

    Save the final report as /research_report.md.
    """

    print("\nStarting research workflow...\n")

    result = research_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": research_task
            }
        ]
    })

    print("\nResearch response:")

    print(result["messages"][-1].content)

    inspect_tool_calls(result)

    report_path = SANDBOX_DIR / "research_report.md"

    if not report_path.exists():
        raise FileNotFoundError(
            "Research report was not created."
        )

    print("\nResearch report generated:", report_path)


# =====================================================
# 11. PRESENTATION WORKFLOW
# =====================================================

def run_presentation():

    presentation_task = """
    Read /research_report.md.

    Delegate the presentation creation task to
    your slide-maker sub-agent.

    The slide-maker must:

    1. Read the research report.
    2. Retrieve the presentation skill.
    3. Extract the important findings.
    4. Create a PowerPoint using create_presentation.

    Save the file as /research_presentation.pptx.
    """

    print("\nStarting presentation workflow...\n")

    result = presentation_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": presentation_task
            }
        ]
    })

    print("\nPresentation response:")

    print(result["messages"][-1].content)

    inspect_tool_calls(result)

    presentation_path = (
        SANDBOX_DIR / "research_presentation.pptx"
    )

    if not presentation_path.exists():
        raise FileNotFoundError(
            "PowerPoint presentation was not created."
        )

    print("\nPresentation generated:", presentation_path)


# =====================================================
# 12. MAIN EXECUTION
# =====================================================

if __name__ == "__main__":

    topic = input(
        "\nEnter a topic for research and presentation: "
    ).strip()

    if not topic:
        raise ValueError("Topic cannot be empty")

    # Step 1: Research
    run_research(topic)

    # Step 2: Generate presentation
    run_presentation()

    print("\n====================================")
    print("APPLICATION COMPLETED SUCCESSFULLY")
    print("====================================")

    print("\nGenerated files:")

    print("- sandbox/research_report.md")

    print("- sandbox/research_presentation.pptx")
