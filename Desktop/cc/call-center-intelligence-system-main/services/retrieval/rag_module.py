import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY") or ""
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME") or ""

# Initialize the embedding model (Turkish-compatible multilingual model, outputs 384 dimensions)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_embeddings = None
_vector_store = None

def _get_embeddings():
    """Lazy-load embedding model (Singleton pattern). Loads into memory only when called."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings

def health_check() -> dict:
    """
    Checks if Pinecone configuration is valid and the index is reachable.
    Returns:
        dict: {"status": "ok" | "error", "message": str}
    """
    if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here":
        return {
            "status": "error",
            "message": "PINECONE_API_KEY eksik veya varsayılan değerde kalmış. Lütfen .env dosyasını kontrol edin."
        }
    if not PINECONE_INDEX_NAME:
        return {
            "status": "error",
            "message": "PINECONE_INDEX_NAME tanımlı değil. Lütfen .env dosyasını kontrol edin."
        }
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_names = [idx.name for idx in pc.list_indexes()]
        if PINECONE_INDEX_NAME not in index_names:
            return {
                "status": "error",
                "message": f"'{PINECONE_INDEX_NAME}' adında bir Pinecone indeksi bulunamadı. Mevcut indeksler: {index_names}"
            }
        return {
            "status": "ok",
            "message": f"Pinecone bağlantısı başarılı! İndeks: '{PINECONE_INDEX_NAME}' hazır."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Pinecone bağlantı doğrulaması başarısız: {e}"
        }

def _get_vector_store():
    """Lazy-load PineconeVectorStore (Singleton pattern). Reuses existing connection instance."""
    global _vector_store
    if _vector_store is None:
        if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here":
            raise ValueError(
                "PINECONE_API_KEY eksik veya placeholder ('your_pinecone_api_key_here') olarak kalmış. "
                "Lütfen geçerli API anahtarınızı .env dosyasına ekleyin."
            )
        if not PINECONE_INDEX_NAME:
            raise ValueError(
                "PINECONE_INDEX_NAME tanımlanmamış. "
                "Lütfen hedef indeks adınızı .env dosyasına ekleyin."
            )

        _vector_store = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=_get_embeddings()
        )
    return _vector_store

# Absolute project root directory and default policy path
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_POLICY_FILE = os.path.join(_BASE_DIR, "data", "sirket_politikalari.md")

def index_policies(file_path: str = DEFAULT_POLICY_FILE):
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

    # Clear existing vectors from Pinecone before uploading new ones
    print(f"[RAG] Connecting to Pinecone index '{PINECONE_INDEX_NAME}' to clear old records...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(str(PINECONE_INDEX_NAME))
        index.delete(delete_all=True)
        print("[RAG] Old records deleted successfully.")
    except Exception as e:
        print(f"[RAG] Note: Could not clear old index records (might be empty already): {e}")

    # Upload chunks to Pinecone
    print(f"[RAG] Uploading vector embeddings to Pinecone index: {PINECONE_INDEX_NAME}...")
    vector_store = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=_get_embeddings(),
        index_name=PINECONE_INDEX_NAME
    )
    print("[RAG] Indexing completed successfully!")

def ensure_indexed():
    """
    Checks if the Pinecone index contains vectors.
    If the index is empty (0 vectors), automatically runs index_policies()
    so the system doesn't return empty context silently.
    """
    if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here" or not PINECONE_INDEX_NAME:
        return
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(str(PINECONE_INDEX_NAME))
        stats = index.describe_index_stats()
        total_vectors = getattr(stats, "total_vector_count", 0)
        if total_vectors == 0:
            print("[RAG INFO] Pinecone indeksi boş tespit edildi. Şirket politikaları otomatik olarak indeksleniyor...")
            index_policies()
    except Exception as e:
        print(f"[RAG WARNING] İndeks doluluk kontrolü yapılamadı: {e}")

def retrieve_context(query: str, k: int = 3) -> str:
    """
    Searches the Pinecone Vector Database for the top 'k' most relevant policy details
    based on the input query. Returns them concatenated as a single context string.
    """
    if not query or not query.strip():
        return ""

    try:
        # Auto-bootstrap: İndeks boşsa aramadan önce otomatik indeksle
        ensure_indexed()

        vector_store = _get_vector_store()
        
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
