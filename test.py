from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
embedder = OllamaEmbeddings(model="mxbai-embed-large")

query = "What is gradient descent?"
query_vector = embedder.embed_query(query)

results = client.query_points(
    collection_name="ml_textbooks",
    query=query_vector,
    limit=3
)
print(results)
# for r in results:
#     print(f"\n📘 {r.payload.get('textbook', 'Unknown Textbook')} [score={r.score:.4f}]")
#     print(r.payload.get('text')[:400], "...")
