import json
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI #if use chatgpt model
from langchain_anthropic import ChatAnthropic #if use claund model
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tools
from tools import search_tools, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide on other text.\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tools, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("what can I help you research? ")
raw_response = agent_executor.invoke({"query": query})

try:
    output_text = raw_response.get("output", "").strip()
    if output_text.startswith("```json") and output_text.endswith("```"):
        output_text = output_text[7:-3].strip()
    if output_text.startswith("{") and output_text.endswith("}"):
        parsed_data = json.loads(output_text)
        structured_response = parser.parse(json.dumps(parsed_data))
        print("Structured Response:", structured_response)
    else:
        print("⚠️ Warning: Output is not valid JSON:", output_text)
except json.JSONDecodeError as jde:
    print("Error parsing JSON:", jde)
except Exception as e:
    print("❌ Error parsing response:", e, "\n🔹 Raw Response -", raw_response)