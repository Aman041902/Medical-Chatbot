from dotenv import load_dotenv
import os
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
HF_API_KEY = os.environ.get('HF_API_KEY')

# Set as environment variable for LangChain compatibility
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# 1. Load and process PDF data
extracted_data = load_pdf_file(data='data/')
filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered_data)

# 2. Load HF embeddings
embeddings = download_hugging_face_embeddings(api_key=HF_API_KEY)

# 3. Setup Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-bot"

# Create index if not exists
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # based on e5-small-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# 4. Initialize Pinecone vector store and upload data
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
