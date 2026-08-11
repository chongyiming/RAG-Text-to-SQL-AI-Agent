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
from sentence_transformers import CrossEncoder, SentenceTransformer
import chromadb

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

llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash",google_api_key=os.environ["AGENT_1_GEMINI_KEY"])

embedding_model=SentenceTransformer("shibing624/text2vec-base-chinese")
chromadb_client=chromadb.PersistentClient("./chroma_db")
chromadb_collection=chromadb_client.get_or_create_collection(name="default")

def split_into_chunks(doc_file:str)->list:
    with open(doc_file, "r", encoding="utf-8") as file:
        content=file.read()
    return [chunk for chunk in content.split("\n\n")]

def embed_chunk(chunk:str)->list:
    embedding=embedding_model.encode(chunk)
    return embedding.tolist()

def sync_knowledge_base():
    chunks=split_into_chunks("doc.txt")
    embeddings=[embed_chunk(chunk) for chunk in chunks]
    
    def save_embeddings(chunks:list,embeddings:list)->None:
        ids=[str(i) for i in range(len(embeddings))]
        chromadb_collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
    save_embeddings(chunks,embeddings)
sync_knowledge_base()

if st.button("Sync Knowledge Base"):
  try:
    chromadb_client.delete_collection(name="default")
    chromadb_collection = chromadb_client.get_or_create_collection(name="default")
    sync_knowledge_base()
  except Exception:
    pass




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

@tool(description="Retrieve the top-k most semantically similar documents using embedding-based vector search.")
def retrieve(query:str, top_k:int)->list:
    query_embedding=embed_chunk(query)
    results=chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]




@tool(description="Rerank retrieved chunks using a cross-encoder that scores query-document relevance, then return the top-k most relevant chunks.")
def rerank(query:str, retrieved_chunks:list,top_k:int)->list:
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs=[(query,chunk) for chunk in retrieved_chunks]
    scores=cross_encoder.predict(pairs)
    chunk_with_score_list=[(chunk,score) for chunk, score in zip(retrieved_chunks,scores)]
    chunk_with_score_list.sort(key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in chunk_with_score_list][:top_k]



agent = create_agent(
    model=llm,
    # tools=[get_schema, run_query],
    tools=[retrieve, rerank,get_schema, run_query],
    checkpointer=checkpointer,
    system_prompt = """ You are a SQL Server and knowledge retrieval expert. Your goal is to answer the user's question accurately using the most appropriate available tools. Workflow: 1. Understand the user's request and determine what type of information is needed. 2. Decide whether the answer should come from: - SQL Server: when the user asks for structured, transactional, or database data. - Vector database / knowledge base: when the user asks about documentation, descriptions, explanations, business rules, or unstructured information. - Both: when the question requires combining database data with information from the knowledge base. 3. If SQL Server is required: - Inspect the database schema if you haven't already. - Never invent tables or columns. - Generate valid SQL Server SQL. - Only generate SELECT statements. Never modify database data. - Execute the query using run_query. 4. If the vector database is required: - Search the knowledge base using the appropriate retrieval tool. - Use the retrieved information to answer the user's question. 5. If both sources are required: - Retrieve relevant knowledge from the vector database. - Query SQL Server for the required structured data. - Combine the information to produce the final answer. 6. If the available information is insufficient, clearly state what information is missing instead of guessing. Final response: - If SQL Server was queried, ALWAYS include the exact SQL query that was executed in a ```sql code block before showing the results. - If no SQL query was executed, do not include an SQL code block. - Present results in a clear and readable format. - Do not expose internal reasoning or tool-selection decisions. Important: - Never invent database tables, columns, values, or knowledge. - Use retrieved information as the source of truth. - Do not execute INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or other write operations. """,
)



st.title("AI-Powered Knowledge Base & Text-to-SQL Agent")
st.divider()


if "message" not in st.session_state:
    st.session_state["message"]=[{"role":"assistant","content":"How can I help?"}]

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input(
    "Ask question about your data or knowledge base..."
)


if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({
        "role": "user",
        "content": prompt,
    })

    with st.spinner():

        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]},config=config)
        st.chat_message("assistant").write(result["messages"][-1].content[0]["text"])
        st.session_state["message"].append({"role":"assistant","content":result["messages"][-1].content[0]["text"]})