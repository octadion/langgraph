import sqlparse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate 
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_openai_tools_agent, AgentExecutor, create_sql_agent
from database.db import get_schema, db
from utils.prompt import SQL_PROMPT, SQL_RESPONSE_QUERY, POSTGRES_PROMPT, full_prompt

def __parse_sql(inp):
    comps = inp.split("[SQL]")
    res = inp
    if comps:
        res = sqlparse.format(comps[-1], reindent=True)
    return res.replace("```sql", "").replace("```", "")

def build_sql_chain(sql_llm, chat_model):
    prompt = ChatPromptTemplate.from_template(SQL_PROMPT)

    sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | sql_llm.bind(stop=["\nSQL Query:"])
        | StrOutputParser()
        | RunnableLambda(__parse_sql)
    )

    prompt_response = PromptTemplate.from_template(SQL_RESPONSE_QUERY)
    full_chain = (
        RunnablePassthrough.assign(query=sql_response).assign(
            schema=get_schema,
            response=lambda x: db.run(x["query"]),
        )
        | prompt_response
        | chat_model
        | StrOutputParser()
    )

    return full_chain

def build_openai_sql(llm):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    context = toolkit.get_context()
    tools = toolkit.get_tools()
    
    prompt = ChatPromptTemplate.from_template(POSTGRES_PROMPT)
    prompt = full_prompt.partial(**context)    
    agent = create_openai_tools_agent(llm, tools, prompt)
    # agent = create_sql_agent(llm, toolkit=toolkit,agent_type="openai-tools")
    
    return AgentExecutor(agent=agent,
                        tools=toolkit.get_tools(),
                        verbose=True,)