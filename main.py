from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_ollama import ChatOllama
from langchain.tools import tool
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

SQL_SERVER = os.environ["SQL_SERVER"]
SQL_USERNAME = quote_plus(os.environ["SQL_USERNAME"])
SQL_PASSWORD = quote_plus(os.environ["SQL_PASSWORD"])
SQL_DATABASE = os.environ["SQL_DATABASE"]

uri = (
    f"mssql+pyodbc://{SQL_USERNAME}:{SQL_PASSWORD}@{SQL_SERVER}:1434/{SQL_DATABASE}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
)
db = SQLDatabase.from_uri(uri)

# llm = ChatOllama(model="qwen3.5:2b")
llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# checkpointer=InMemorySaver()
connection=sqlite3.connect("resources/checkpoint.db",check_same_thread=False)
checkpointer=SqliteSaver(connection)
checkpointer.setup()

config={"configurable":{"thread_id":"thread_1"}}

@tool(description="Get the database schema and sample rows.")
def get_schema() -> str:
    return db.table_info

@tool(description="Run a SQL Server SELECT query against the database and return the results.")
def run_query(query: str) -> str:
    return QuerySQLDataBaseTool(db=db).invoke(query)

agent = create_agent(
    model=llm,
    tools=[get_schema, run_query],
    checkpointer=checkpointer,
    system_prompt="""
    You are a SQL Server expert.

    Workflow:
    1. Understand the user's request.
    2. Inspect the schema if you haven't already.
    3. Generate valid SQL Server SQL (SELECT only).
    4. Execute the query using run_query.
    5. Return the results in a readable format.

    IMPORTANT: In your final response, ALWAYS include the exact SQL query you executed,
    formatted in a ```sql code block, before showing the results table.

    Never invent tables or columns.
    """,
)


st.title("Text to SQL AI Agent")
st.divider()


if "message" not in st.session_state:
    st.session_state["message"]=[{"role":"assistant","content":"How can I help?"}]

if "agent" not in st.session_state:
    st.session_state["agent"]=agent

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input(
    "Ask your database a question...",
)


if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({
        "role": "user",
        "content": prompt,
    })

    with st.spinner():

        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]},config=config)
        print(result)
        st.chat_message("assistant").write(result["messages"][-1].content[0]["text"])
        st.session_state["message"].append({"role":"assistant","content":result["messages"][-1].content[0]["text"]})