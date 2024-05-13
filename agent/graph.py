import functools
import operator
from typing import Annotated, List, TypedDict
from tools.tools import build_utility_tools, build_rag_tools, build_search_tools
from langchain_core.messages import BaseMessage, HumanMessage
from agent.agent import build_openai_sql
from agent.multi_agent import create_agent, create_team_supervisor, agent_node, agent_with_chain
from tools.sql_tool import SQLTool
from utils.prompt import TEAM_SUPERVISOR_PROMPT, TOP_SUPERVISOR_PROMPT
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph

class ResearchTeamState(TypedDict):

    messages: Annotated[List[BaseMessage], operator.add]

    team_members: List[str]

    next: str

class DataTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str
        
class SummaryTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    if "FINISH" in last_message.content:
        return "FINISH"
    else:
        return state["next"]

def build_research_team(chat_model):
    util_tools = build_utility_tools(chat_model)
    tools = util_tools + build_search_tools()
    search_agent = create_agent(
        chat_model,
        tools,
        "You are a research assistant who can search for valid info or data related to komunitas anggota domain"
         "Remember to pick a proper tool to use based on the tool's description.",
    )
    search_node = functools.partial(agent_node, agent=search_agent, name="Search")

    supervisor_agent = create_team_supervisor(
        chat_model,
        TEAM_SUPERVISOR_PROMPT,
        ["Search"],
    )

    research_graph = StateGraph(ResearchTeamState)
    research_graph.add_node("Search", search_node)
    research_graph.add_node("supervisor", supervisor_agent)

    research_graph.add_edge("Search", "supervisor") 
    research_graph.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "Search": "Search",
            "FINISH": END
        },
    )

    research_graph.set_entry_point("supervisor")
    chain = research_graph.compile()

    def enter_chain(message: str,  members: List[str]):
        results = {
            "messages": [HumanMessage(content=message)],
            "team_members": ", ".join(members),
        }
        return results

    return (functools.partial(enter_chain, members=research_graph.nodes) | chain)

def build_data_team(sql_llm, chat_model):
    util_tools = build_utility_tools(chat_model)
    
    runnable_sql = build_openai_sql(sql_llm)
    sql_tool = SQLTool(sql_chain=runnable_sql, handle_tool_error=True)
    sql_agent = create_agent(
        chat_model,
        util_tools + [sql_tool] + build_search_tools(),
        "You are an useful data assistant. You are responsible for requests of retrieving user or system data from the database by using provided tools."
        "\nYou will return the raw data.",
    )
    sql_node = functools.partial(agent_node, agent=sql_agent, name="SQL")
    
    # Build agent for RAG with internal data
    retriever_tools = build_rag_tools(chat_model)
    retriever_agent = create_agent(
        chat_model,
        util_tools + retriever_tools,
        "You are a komunitas anggota assistant who can retrieve information from the embedding documents related to komunitas anggota information",
    )
    retriever_node = functools.partial(agent_node, agent=retriever_agent, name="RAG")

    supervisor_agent = create_team_supervisor(
        chat_model,
        TEAM_SUPERVISOR_PROMPT,
        ["SQL", "RAG"],
    )

    data_graph = StateGraph(DataTeamState)
    data_graph.add_node("SQL", sql_node)
    data_graph.add_node("RAG", retriever_node)
    data_graph.add_node("supervisor", supervisor_agent)

    # control flow
    data_graph.add_edge("SQL", "supervisor")
    data_graph.add_edge("RAG", "supervisor")
    data_graph.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "SQL": "SQL",
            "RAG": "RAG",
            "FINISH": END
        },
    )

    data_graph.set_entry_point("supervisor")
    chain = data_graph.compile()

    def enter_chain(message: str,  members: List[str]):
        results = {
            "messages": [HumanMessage(content=message)],
            "team_members": ", ".join(members),
        }
        return results

    return (functools.partial(enter_chain, members=data_graph.nodes) | chain)

def build_summary_team(llm):
    util_tools = build_utility_tools(llm)
    
    summary_agent = create_agent(
        llm,
        util_tools,
        "You are a komunitas anggota analysis and summarization expert."
        "You are responsible to write a concise summary for questions related to users' question about komunitas anggota or a needed summary by using provided tools based on the inputting data."
        "The summary MUST include all received data."
        "Remember that if a summary about question of komunitas anggota is built, if the summary is good, give encourages to engage him to keep it up appended at the end of the summary. Otherwise, if it is not good, give advice to help him improve."
    )
    summary_node = functools.partial(agent_node, agent=summary_agent, name="Summarization")
    
    supervisor_agent = create_team_supervisor(
        llm,
        TEAM_SUPERVISOR_PROMPT,
        ["Summarization"],
    )

    summary_graph = StateGraph(SummaryTeamState)
    summary_graph.add_node("Summarization", summary_node)
    summary_graph.add_node("supervisor", supervisor_agent)

    summary_graph.add_edge("Summarization", "supervisor")
    summary_graph.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "Summarization": "Summarization",
            "FINISH": END
        },
    )

    summary_graph.set_entry_point("supervisor")
    chain = summary_graph.compile()

    def enter_chain(message: str,  members: List[str]):
        results = {
            "messages": [HumanMessage(content=message)],
            "team_members": ", ".join(members),
        }
        return results

    return (functools.partial(enter_chain, members=summary_graph.nodes) | chain)

def build_general_team(chat_model):
    general_agent = create_agent(
        chat_model,
        build_utility_tools(chat_model),
        "You are a helpful assistant mainly focused on komunitas anggota domain. You are responsible for answering general questions.",
    )
    general_node = functools.partial(agent_node, agent=general_agent, name="General")

    supervisor_agent = create_team_supervisor(
        chat_model,
        TEAM_SUPERVISOR_PROMPT,
        ["General"],
    )

    general_graph = StateGraph(ResearchTeamState)
    general_graph.add_node("General", general_node)
    general_graph.add_node("supervisor", supervisor_agent)

    general_graph.add_edge("General", "supervisor")
    general_graph.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "General": "General",
            "FINISH": END
        },
    )

    general_graph.set_entry_point("supervisor")
    chain = general_graph.compile()

    def enter_chain(message: str,  members: List[str]):
        results = {
            "messages": [HumanMessage(content=message)],
            "team_members": ", ".join(members),
        }
        return results

    return (functools.partial(enter_chain, members=general_graph.nodes) | chain)

class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


def get_last_message(state: State) -> str:
    return state["messages"][-1].content


def join_graph(response: dict):
    content = [response["messages"][-1]]
    print(f"Tracking join graph: {content}")
    return {"messages": content}

def build_supervisor(sql_llm, chat_model, memory = None) -> CompiledGraph:
    supervisor_node = create_team_supervisor(
        chat_model,
        TOP_SUPERVISOR_PROMPT,
        ["Research Team", "Data Team", "Summary Team", "General Team"],
    )

    super_graph = StateGraph(State)

    research_chain = build_research_team(chat_model)
    super_graph.add_node("Research Team", get_last_message | research_chain | join_graph)

    data_chain = build_data_team(sql_llm, chat_model)
    super_graph.add_node(
        "Data Team", get_last_message | data_chain | join_graph
    )
    
    summary_chain = build_summary_team(chat_model)
    super_graph.add_node(
        "Summary Team", get_last_message | summary_chain | join_graph
    )

    general_chain = build_general_team(chat_model)
    super_graph.add_node(
        "General Team", get_last_message | general_chain | join_graph
    )

    super_graph.add_node("supervisor", supervisor_node)

    super_graph.add_edge("Research Team", "supervisor")
    super_graph.add_edge("Data Team", "supervisor")
    super_graph.add_edge("Summary Team", "supervisor")    
    super_graph.add_edge("General Team", "supervisor")
    super_graph.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "Research Team": "Research Team",
            "Data Team": "Data Team",
            "Summary Team": "Summary Team",
            "General Team": "General Team",
            "FINISH": END,
        },
    )
    super_graph.set_entry_point("supervisor")

    if memory:
        super_graph = super_graph.compile(checkpointer=memory)
    else:
        super_graph = super_graph.compile()

    return super_graph