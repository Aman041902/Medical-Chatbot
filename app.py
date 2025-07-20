from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpoint
from src.helper import download_hugging_face_embeddings
from src.prompt import *

import os

app = Flask(__name__)
load_dotenv()

# Load environment variables
os.environ["PINECONE_API_KEY"] = os.environ.get("PINECONE_API_KEY")
os.environ["HF_API_KEY"] = os.environ.get("HF_API_KEY")

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Load Pinecone vector store
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-bot",
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Use Hugging Face LLM 
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.5,
    max_new_tokens=256
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Chain setup
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    user_msg = request.form["msg"]
    response = rag_chain.invoke({"input": user_msg})
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
