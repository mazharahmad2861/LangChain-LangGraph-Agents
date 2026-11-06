
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend


# =====================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env", override=True)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing")

if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY is missing")


# =====================================================
# 2. SETUP MODEL
# =====================================================

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
    temperature=0
)


# =====================================================
# 3. SETUP SEARCH TOOL
# =====================================================

search = GoogleSerperRun(
    api_wrapper=GoogleSerperAPIWrapper()
)


# =====================================================
# 4. SETUP SANDBOX
# =====================================================

sandbox = os.path.abspath(
    PROJECT_DIR / "sandbox"
)

os.makedirs(sandbox, exist_ok=True)

backend = FilesystemBackend(
    root_dir=sandbox,
    virtual_mode=True
)


# =====================================================
# 5. SYSTEM PROMPT
# =====================================================

system_prompt = """
You are an expert AI research analyst.

Your responsibility is to conduct detailed research
and prepare structured Markdown reports.

For every research task:

1. Understand the user's research objective.
2. Create a plan using the todo tool.
3. Break the topic into smaller research questions.
4. Use the search tool to gather information.
5. Prefer reliable and relevant sources.
6. Compare information from multiple sources.
7. Identify important findings and insights.
8. Mention sources or URLs in the report where possible.
9. Write a structured Markdown report.

Do not simply return search results.

Analyze and organize the information.

Use the filesystem tools to save the final report.
"""


# =====================================================
# 6. CREATE DEEP AGENT
# =====================================================

research_agent = create_deep_agent(
    model=model,
    tools=[search],
    system_prompt=system_prompt,
    backend=backend
)


# =====================================================
# 7. GET RESEARCH TOPIC
# =====================================================

topic = input("\nEnter your research topic: ")

if not topic.strip():
    raise ValueError("Research topic cannot be empty")


# =====================================================
# 8. CREATE USER TASK
# =====================================================

research_task = f"""
Conduct research on the following topic:

{topic}

Prepare a comprehensive but concise report.

Include these sections:

1. Introduction
2. Current Landscape
3. Major Developments
4. Applications
5. Challenges
6. Opportunities
7. Future Trends
8. Conclusion
9. Sources

Use your todo tool to plan the research.

Use web search to collect information.

Save the final report as:

/research_report.md
"""


# =====================================================
# 9. EXECUTE AGENT
# =====================================================

print("\nResearch agent started...\n")

result = research_agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": research_task
        }
    ]
})


# =====================================================
# 10. DISPLAY FINAL RESPONSE
# =====================================================

print("\n========== AGENT RESPONSE ==========\n")

print(result["messages"][-1].content)


# =====================================================
# 11. INSPECT TOOL CALLS
# =====================================================

tools_used = []

for message in result["messages"]:

    tool_calls = getattr(message, "tool_calls", []) or []

    for tool_call in tool_calls:
        tools_used.append(tool_call["name"])


print("\n========== TOOLS USED ==========\n")

print(tools_used)


# =====================================================
# 12. CHECK GENERATED REPORT
# =====================================================

report_path = Path(sandbox) / "research_report.md"

if report_path.exists():
    print("\nReport successfully generated!")
    print("Location:", report_path)
else:
    print("\nReport was not generated.")
