"""
upload_to_qdrant.py
One-time scalable script to upload all textbook chunks to Cloud Qdrant.
"""

import os
import json
import logging
from glob import glob
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from langchain_ollama import OllamaEmbeddings  # or use another embeddings provider
from dotenv import load_dotenv
import uuid
load_dotenv()

# ---------- CONFIG ----------
OUTPUT_DIR = "../outputs"
COLLECTION_NAME = "ml_textbooks"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# embedding model
EMBED_MODEL = "mxbai-embed-large"
BATCH_SIZE = 64

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qdrant_uploader")


# ---------- FUNCTIONS ----------
def get_all_chunks() -> list:
    """Load all chunks.jsonl files from outputs/ with progress bar"""
    all_chunks = []
    chunk_paths = glob(os.path.join(OUTPUT_DIR, "*/chunks.jsonl"))
    for path in tqdm(chunk_paths, desc="📖 Loading all chunks.jsonl files", unit="file"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    logger.info(
        "Loaded %d chunks from %d books",
        len(all_chunks),
        len(set(p.split('/')[2] for p in chunk_paths))
    )
    return all_chunks


def ensure_collection(client: QdrantClient, dim: int):
    """Create collection if not exists"""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        logger.info("Created new collection '%s' with dim=%d", COLLECTION_NAME, dim)
    else:
        logger.info("Collection '%s' already exists", COLLECTION_NAME)


def batch_upload(client: QdrantClient, chunks: list, embedder):
    """Embed and upload chunks to Qdrant with progress bars"""
    texts = [c["text"] for c in chunks]
    embeddings = []

    # progress for embedding generation
    for i in tqdm(range(0, len(texts), 8), desc="🔍 Generating embeddings", unit="batch"):
        sub_texts = texts[i:i+8]
        try:
            embeddings.extend(embedder.embed_documents(sub_texts))
        except Exception as e:
            logger.error("Embedding failed at batch %d: %s", i, e)
            continue

    points = []
    for i, emb in enumerate(tqdm(embeddings, desc="🧩 Building Qdrant points", unit="vec")):
        meta = chunks[i]["metadata"]
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),  # Qdrant auto-assigns
                vector=emb,
                payload=meta
            )
        )


    # upload with progress
    for i in tqdm(range(0, len(points), 32), desc="☁️ Uploading to Qdrant", unit="batch"):
        sub_points = points[i:i+32]
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=sub_points)
        except Exception as e:
            logger.error("Upload failed for batch %d: %s", i, e)
            continue


def main():
    if not QDRANT_API_KEY or "YOUR_QDRANT_API_KEY" in QDRANT_API_KEY:
        raise ValueError("❌ Set QDRANT_API_KEY env var before running.")
    if not QDRANT_URL.startswith("https://"):
        raise ValueError("❌ Invalid QDRANT_URL, must start with https://")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
    embedder = OllamaEmbeddings(model=EMBED_MODEL)

    all_chunks = get_all_chunks()
    if not all_chunks:
        logger.error("No chunks found in outputs/. Exiting.")
        return

    # warmup embedding to detect vector size
    dim = len(embedder.embed_documents(["dimension_check"])[0])
    ensure_collection(client, dim)

    logger.info("Starting upload to Qdrant Cloud (%s)...", COLLECTION_NAME)
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="🚀 Uploading batches", unit="batch"):
        batch = all_chunks[i:i + BATCH_SIZE]
        try:
            batch_upload(client, batch, embedder)
        except Exception as e:
            logger.exception("Upload failed for batch %d: %s", i // BATCH_SIZE, e)
            continue

    logger.info("✅ Completed upload of %d chunks to '%s'", len(all_chunks), COLLECTION_NAME)


if __name__ == "__main__":
    main()
