from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from utils.constants import prompt

SQL_PROMPT="""
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- NEVER self-assuming any value, instead, you should join tables to get relevant data or key constraints.
- Remember that users' komunitas anggota are located in the `anggota` table entirely.
- Eventually, if you cannot answer the question with the available database schema, return 'I do not know'.

### Database Schema
The query will run on a database with the following schema:
{schema}

### Answer
SQL Query: [SQL]
"""
SQL_RESPONSE_QUERY="""
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- NEVER self-assuming any value, instead, you should join tables to get relevant data or key constraints.
- Remember that users' komunitas anggota are located in the `anggota` table entirely.
- Eventually, if you cannot answer the question with the available database schema, return 'I do not know'.

### Database Schema
The query will run on a database with the following schema:
{schema}

### Answer
SQL Query: {query}
SQL Response: {response}
"""

POSTGRES_PROMPT = """You are an agent designed to interact with a SQL database.
Given an input question in Indonesian language, create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
    
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
If the question does not seem related to the database, just return "I don't know" as the answer.
If no data is found, respond with "No data found".

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"

Only use the following tables:
{table_info}

Question: {question}

{agent_scratchpad}

Here are some examples of user inputs and their corresponding SQL queries:
"""

TEAM_SUPERVISOR_PROMPT="""You are a supervisor tasked with managing a conversation between the following workers:  {team_members}.
Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. FINISH the worker IF already found appropiate ANSWER!
When finished, respond with FINISH.
"""

TOP_SUPERVISOR_PROMPT="""You are a supervisor tasked with managing a conversation between the following workers:  {team_members}.
Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. FINISH the worker IF already found appropiate ANSWER!
If the question is related to a person's komunitas anggota data, you should route to the worker that processes with the database as the first choice.
When finished, respond with FINISH.
"""
prompt_selector = SemanticSimilarityExampleSelector.from_examples(
        prompt,
        OpenAIEmbeddings(),
        FAISS,
        k=5,
        input_keys=["question"],
    )

few_shot_prompt = FewShotPromptTemplate(
        example_selector=prompt_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {question}\nSQL query: {query}"
        ),
        input_variables=["question", "dialect", "top_k"],
        prefix=POSTGRES_PROMPT,
        suffix="",
    )
    
full_prompt = ChatPromptTemplate.from_messages(
        [
            # MessagesPlaceholder(variable_name="history"),
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{question}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )