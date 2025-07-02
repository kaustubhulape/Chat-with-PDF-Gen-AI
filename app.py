This is Chat with PDF Project Code. Give me line to line explanation. import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# chat template to be used
template = """
You are an assistance bot for a PDF file. 
Answer the user's question using the context provided.
If the question can not be answered using the context, return "I dont know".
Question: {question}
Context: {context}
Answer:
"""

# create the model
model = ChatOllama(model="llama3")

def create_embeddings_for_pdf_file(pdf_file):

    # load the file
    reader = PdfReader(pdf_file)

    # read the data
    pages = reader.pages
    print(pages)

    # collect the documents
    docs = []
    id = 1
    for page in pages:
        contents = page.extract_text()
        docs.append(Document(id=id, page_content=contents))
        id += 1
    
    # split the documents using splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    docs = splitter.split_documents(docs)

    # create the chroma vector store
    vector_store = Chroma(
        embedding_function=OllamaEmbeddings(model="llama3"),
        persist_directory="./chroma_db",
        collection_name="pdf_collection"
    )
    
    # add the documents
    vector_store.add_documents(docs)

    # store the vector_store in session state
    st.session_state.vector_store = vector_store

def get_answer_of_user_question(question):
    # get the context
    context = st.session_state.vector_store.search(question, search_type="similarity", k=5)

    # create the prompt
    prompt_template = ChatPromptTemplate.from_template(template=template)
    prompt = prompt_template.invoke({"question": question, "context": context})

    # get the answer
    response = model.invoke(prompt)
    st.write(response.content)


st.title("Chat with PDFs")

# show the sidebar
with st.sidebar:
    st.subheader("PDF files")

    # get the pdf file uploaded from user
    pdf_file = st.file_uploader("upload your pdf file", type="pdf")

    # if pdf file is uploaded
    if pdf_file:
        create_embeddings_for_pdf_file(pdf_file)

# check if the vector store is created
if 'vector_store' in st.session_state:
    st.write("PDF file uploaded and embeddings created")

# get the question from user
question = st.chat_input("Ask your question about the pdf file")

if question:
    # get the answer
    get_answer_of_user_question(question)
    