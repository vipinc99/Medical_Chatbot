import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from dotenv import load_dotenv, find_dotenv
from langchain.docstore.document import Document
load_dotenv(find_dotenv())


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
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
    uploaded_file = st.file_uploader("Upload a PDF file",type="pdf")
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        documents = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:  # Avoid None pages
                documents.append(Document(page_content=text, metadata={"page": i}))
        # Store embeddings in FAISS
        @st.cache_resource
        def get_vectorstore():
            def create_chunks(extracted_data):
                text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                            chunk_overlap=50)
                text_chunks=text_splitter.split_documents(extracted_data)
                return text_chunks
            text_chunks=create_chunks(extracted_data=documents)
            def get_embedding_model():
                embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                return embedding_model
            embedding_model=get_embedding_model() 
            db=FAISS.from_documents(text_chunks, embedding_model)
            return db
        
        if 'messages2' not in st.session_state:
            st.session_state.messages2 = []

        for message in st.session_state.messages2:
            st.chat_message(message['role']).markdown(message['content'])

        prompt=st.chat_input("Pass your prompt here")

        if prompt:
            st.chat_message('user').markdown(prompt)
            st.session_state.messages2.append({'role':'user', 'content': prompt})

            CUSTOM_PROMPT_TEMPLATE = """
                    Use the pieces of information provided in the context to answer user's question.
                    If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                    Dont provide anything out of the given context

                    Context: {context}
                    Question: {question}

                    Start the answer directly. No small talk please.
                    """
            
            HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
            HF_TOKEN=os.environ.get("HF_TOKEN")

            try: 
                vectorstore=get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")

                qa_chain=RetrievalQA.from_chain_type(
                    llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response=qa_chain.invoke({'query':prompt})

                result=response["result"]
                # source_documents=response["source_documents"]
                result_to_show=result
                #response="Hi, I am MediBot!"
                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages2.append({'role':'assistant', 'content': result_to_show})

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()