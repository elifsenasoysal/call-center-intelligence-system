import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize the embedding model (Turkish-compatible multilingual model, outputs 384 dimensions)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def index_policies(file_path: str = "data/sirket_politikalari.md"):
    """
    Reads the company policies file, splits it into semantic chunks,
    and uploads it to the Pinecone Vector Database.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Policy file not found at: {file_path}")

    print(f"[RAG] Reading policy file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(f"[RAG] Created {len(chunks)} chunks from document.")

    # Upload chunks to Pinecone
    print(f"[RAG] Uploading vector embeddings to Pinecone index: {PINECONE_INDEX_NAME}...")
    vector_store = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print("[RAG] Indexing completed successfully!")

def retrieve_context(query: str, k: int = 3) -> str:
    """
    Searches the Pinecone Vector Database for the top 'k' most relevant policy details
    based on the input query. Returns them concatenated as a single context string.
    """
    try:
        vector_store = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        
        # Perform similarity search
        results = vector_store.similarity_search(query, k=k)
        
        # Concatenate content of top matches
        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        print(f"[RAG] Error during retrieval: {e}")
        return ""

if __name__ == "__main__":
    # If run directly, index the local document
    index_policies()
