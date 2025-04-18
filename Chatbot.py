import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN=os.environ.get("HF_TOKEN")


DB_FAISS_PATH="vectorstore/db_faiss"
       

@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        task="text-generation",
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm


def main():
    st.title("Medical Chatbot!")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
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
                    verbose=False,
                    agent_kwargs={
                        "prefix": """You are a helpful medical assistant. You have access to two tools:
                        - Use the "Internal Medical QA" tool when the user asks something that may be answered from internal hospital or clinical documents.
                        - Use 'PubMed Search' for up-to-date research, recent trials, or when the question mentions "recent", "new", "latest" or "research".

                        Be smart. Do not fabricate answers. Only use one tool per query. Explain your reasoning when appropriate."""
                            }
                )

            response = agent.invoke({"input": prompt})

            result=response["output"]
            # source_documents=response["source_documents"]
            result_to_show=result
            #response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()