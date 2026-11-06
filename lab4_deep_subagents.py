
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend


# =====================================================
# 1. ENVIRONMENT CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env", override=True)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY")

if not os.getenv("SERPER_API_KEY"):
    raise ValueError("Missing SERPER_API_KEY")


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
# 4. SANDBOX AND BACKEND
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
# 5. RESEARCH SUB-AGENT INSTRUCTIONS
# =====================================================

competitor_research_prompt = """
You are a business intelligence research specialist.

You receive the name of one company.

Research that company using the search tool.

Collect information about:

1. Company overview
2. Main products and services
3. Target customers
4. Business model
5. Major strengths
6. Competitive advantages
7. Recent developments

Use reliable sources.

Return concise, structured findings.

Do not research other competitors.

Do not create the final comparison report.
Your responsibility is only to research your assigned company.
"""


# =====================================================
# 6. DEFINE SUB-AGENT
# =====================================================

competitor_researcher = {

    "name": "competitor-researcher",

    "description": (
        "Researches one company and returns "
        "structured business intelligence findings."
    ),

    "system_prompt": competitor_research_prompt,

    "tools": [search]
}


# =====================================================
# 7. LEAD AGENT INSTRUCTIONS
# =====================================================

lead_prompt = """
You are a senior competitive intelligence analyst.

You are responsible for preparing competitor analysis reports.

You have access to a competitor-researcher sub-agent.

For every competitor:

1. Delegate its research to the competitor-researcher.
2. Provide the exact company name.
3. Collect the research findings.
4. Review and organize the findings.

After collecting all research:

1. Compare the companies.
2. Identify similarities and differences.
3. Compare their products and services.
4. Analyze their competitive advantages.
5. Identify market opportunities and threats.
6. Prepare a final comparison report.

Use the todo tool to plan your work.

Save the final report as:

/competitor_analysis.md

Do not invent missing information.
Clearly mention when information could not be verified.
"""


# =====================================================
# 8. CREATE LEAD AGENT
# =====================================================

lead_agent = create_deep_agent(

    model=model,

    tools=[],

    system_prompt=lead_prompt,

    subagents=[competitor_researcher],

    backend=backend
)


# =====================================================
# 9. GET COMPETITORS
# =====================================================

print("\nEnter three competitors to compare.")

company_1 = input("Competitor 1: ").strip()
company_2 = input("Competitor 2: ").strip()
company_3 = input("Competitor 3: ").strip()

companies = [company_1, company_2, company_3]

if any(not company for company in companies):
    raise ValueError("Please enter all three company names")

if len(set(company.lower() for company in companies)) != 3:
    raise ValueError("Please enter three different companies")


# =====================================================
# 10. CREATE MISSION
# =====================================================

mission = f"""
Prepare a detailed competitor analysis report.

Companies to analyze:

1. {company_1}
2. {company_2}
3. {company_3}

Delegate the research of each company to your
competitor-researcher sub-agent.

You should research all three companies.

After receiving all findings, prepare a comparative analysis.

Include:

1. Executive Summary
2. Company Overview
3. Products and Services
4. Target Customers
5. Business Model
6. Competitive Advantages
7. Strengths and Weaknesses
8. Comparison Table
9. Market Opportunities
10. Final Analysis
11. Sources

Save the report as /competitor_analysis.md.
"""


# =====================================================
# 11. EXECUTE LEAD AGENT
# =====================================================

print("\nStarting competitor analysis...\n")

result = lead_agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": mission
        }
    ]
})


# =====================================================
# 12. FINAL RESPONSE
# =====================================================

print("\n========== FINAL REPORT RESPONSE ==========\n")

print(result["messages"][-1].content)


# =====================================================
# 13. TOOL CALL INSPECTION
# =====================================================

tools_used = []

for message in result["messages"]:

    tool_calls = getattr(message, "tool_calls", []) or []

    for tool_call in tool_calls:
        tools_used.append(tool_call["name"])


print("\nTools called by lead agent:")
print(tools_used)


# =====================================================
# 14. CHECK OUTPUT
# =====================================================

report_path = Path(sandbox) / "competitor_analysis.md"

if report_path.exists():

    print("\nCompetitor analysis generated!")
    print("File:", report_path)

else:

    print("\nReport file not found.")
