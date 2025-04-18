import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.tools.pubmed.tool import PubmedQueryRun

# Setup LLM (Mistral with HuggingFace)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        task="text-generation",
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm
    
# Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


# Tool 1: RAG-based QA
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True, #Metadeta
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

internal_doc_tool = Tool(
    name="Internal Medical QA",
    func=lambda q: qa_chain.invoke({"query": q})["result"],  # extract the final answer cleanly
    description="Use this tool for questions answerable from internal hospital or clinical documentation."
)


# Tool 2: PubMed Query
pubmed_tool = PubmedQueryRun()
pubmed_search_tool = Tool(
    name="PubMed Search",
    func=pubmed_tool.run,
    description="Use this for finding the latest research papers, medical trials, and scholarly literature on any medical topic. Especially useful for up-to-date findings, studies, or treatments."
)

tools = [internal_doc_tool, pubmed_search_tool]

agent = initialize_agent(
    tools=tools,
    llm=load_llm(HUGGINGFACE_REPO_ID),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "prefix": """You are a helpful medical assistant. You have access to two tools:
        - Use the "Internal Medical QA" tool when the user asks something that may be answered from internal hospital or clinical documents.
        - Use 'PubMed Search' for up-to-date research, recent trials, or when the question mentions "recent", "new", "latest" or "research".

        Be smart. Do not fabricate answers. Only use one tool per query. Explain your reasoning when appropriate."""
            }
)


# Now invoke with a single query
user_query=input("Write Query Here: ")
response = agent.invoke({"input": user_query})
print("RESULT:\n", response["output"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])