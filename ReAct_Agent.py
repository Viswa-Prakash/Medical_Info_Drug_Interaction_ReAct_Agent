import os
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


llm = init_chat_model("openai:gpt-4.1", temperature=0.7)

#Tools
serper_tool = Tool(
    name = "serper",
    func = GoogleSerperAPIWrapper().run,
    description = "Useful for general medical and drug related search queries"
)
google_search_tool = Tool(
    name = "google_search",
    func = GoogleSearchAPIWrapper().run,
    description = "Use for searching the web for drug safety, symptoms, and health advice"
)
wiki_tool = Tool(
    name = "wiki",
    func = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k=10)).run,
    description = "Use for background and encyclopedic information on drugs or conditions"
)
tavily_tool = Tool(
    name = "tavily",
    func = TavilySearch().run,
    description="Use for summarizing recent web results about drug guidance and health risks."
)
tools = [serper_tool, google_search_tool, wiki_tool, tavily_tool]


react_prompt = """
You are a Medical Information and Drug Interaction Advisor.
Your goals are:
- Identify medicines for symptoms,
- Detect potential drug interactions,
- Suggest usage guidance, side effects, dosages.

Each step, reason step by step and call the BEST tool if needed.

**ALWAYS** end your dialog with:
Final Answer: <summary with drug info, dosage, and safety advice>

Do NOT call more tools after your 'Final Answer:'.
"""

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def reasoning_node(state: State):
    llm_with_tools = llm.bind_tools(tools)
    messages = [{"role": "system", "content": react_prompt}] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": state["messages"] + [response]}

tool_node = ToolNode(tools=tools)

def should_continue(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "content") and "final answer" in last_message.content.lower():
        return "end"
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "action"
    if len(state["messages"]) > 16:
        return "end"
    return "reason"


builder = StateGraph(State)
builder.add_node("reason", reasoning_node)
builder.add_node("action", tool_node)
builder.set_entry_point("reason")
builder.add_conditional_edges(
    "reason",
    should_continue,
    {
        "action": "action",
        "end": END,
    }
)
builder.add_edge("action", "reason")
medical_agent = builder.compile()