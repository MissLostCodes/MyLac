"""
 - Extract text blocks with page coordinates (PyMuPDF)
 - Extract tables (pdfplumber) -> save CSV + text
 - Extract images (PyMuPDF) -> save image files + run OCR (pytesseract)
 - Heuristic tagging for formulas (image OCR containing math characters or bounding box aspects)
 - Chunk text into ~TARGET_TOKENS with overlap
 - Save outputs: outputs/<book_slug>/{chunks.jsonl,manifest.csv,assets/}
 - Optional: embed via OllamaEmbeddings and upsert to Qdrant (commented section)
"""

import os
import json
import csv
import time
import hashlib
import pathlib
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from tqdm import tqdm
from dotenv import load_dotenv

# Optional embedding / vector store imports (uncomment if used)
# from langchain_ollama.OllamaEmbeddings import OllamaEmbeddings
# from qdrant_client import QdrantClient
# from qdrant_client.models import PointStruct

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pdf_chunker")

PDF_DIR = "../../KnowledgeBase"
OUTPUT_DIR = "../outputs"
ASSETS_SUBDIR = "assets"
MAX_PAGE_WORKERS = 6  # concurrency for page extraction
TARGET_TOKENS = 700
TOKEN_CHAR_ESTIMATE = 4
OVERLAP_TOKENS = 75

# Map local filenames to textbook titles
TEXTBOOK_MAP = {
    "applied_ml_gopal.pdf": "Applied Machine Learning by Dr. M. Gopal",
    "intro_ml_alpaydin.pdf": "Introduction to Machine Learning by Ethem Alpaydın",
    "machine_learning_mitchell.pdf": "Machine Learning by Tom M. Mitchell"
}

# helper: safe slug
def safe_slug(s: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in s)
    return keep[:120]

def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / TOKEN_CHAR_ESTIMATE))

def chunk_text(text: str, target_tokens: int = TARGET_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> List[str]:
    # simple paragraph-based chunker with overlap (works well for textbooks)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    cur = ""
    cur_tokens = 0
    for p in paragraphs:
        p_tokens = estimate_tokens(p)
        if cur_tokens + p_tokens <= target_tokens or cur_tokens == 0:
            cur += ("\n\n" if cur else "") + p
            cur_tokens += p_tokens
        else:
            chunks.append(cur.strip())
            overlap_chars = int(overlap_tokens * TOKEN_CHAR_ESTIMATE)
            cur = (cur[-overlap_chars:] + "\n\n" + p).strip()
            cur_tokens = estimate_tokens(cur)
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

# --- PDF extraction helpers ---
def extract_images_from_page(doc: fitz.Document, page_num: int, dest_assets_dir: str, book_slug: str) -> List[Dict[str, Any]]:
    """
    Extract images from a single page using PyMuPDF.
    Returns list of metadata dicts for each image: {asset_path, page, bbox, ocr_text, likely_formula}
    """
    page = doc.load_page(page_num)
    images = page.get_images(full=True)
    extracted = []
    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        ext = base_image.get("ext", "png")
        # unique filename
        h = hashlib.sha1(image_bytes).hexdigest()[:12]
        fname = f"{book_slug}_page{page_num+1}_img{img_index}_{h}.{ext}"
        path = os.path.join(dest_assets_dir, fname)
        with open(path, "wb") as fo:
            fo.write(image_bytes)
        # Try OCR on image to classify (formula vs text)
        try:
            ocr_text = pytesseract.image_to_string(path)
        except Exception as e:
            logger.debug("OCR failed for %s: %s", path, e)
            ocr_text = ""
        # heuristic: if OCR contains math symbols or short symbols, mark formula
        math_chars = set("=+-/<>∑∫√πθαβγλμ^_")  # expand if needed
        likely_formula = any((c in math_chars) for c in ocr_text)
        extracted.append({
            "asset_path": path,
            "page": page_num + 1,
            "ocr_text": ocr_text,
            "likely_formula": likely_formula,
            "width": base_image.get("width"),
            "height": base_image.get("height")
        })
    return extracted

def extract_tables_from_pdf(pdf_path: str, dest_assets_dir: str) -> List[Dict[str, Any]]:
    """
    Use pdfplumber to extract tables per page. Save CSVs and return metadata list.
    """
    out = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables()
                except Exception as e:
                    logger.debug("pdfplumber page %s extract_tables error: %s", i+1, e)
                    tables = None
                if tables:
                    for tindex, table in enumerate(tables):
                        # convert table (list of rows) to CSV
                        safe_name = f"table_page{i+1}_{tindex}.csv"
                        csv_path = os.path.join(dest_assets_dir, safe_name)
                        # sanitize rows and write
                        try:
                            with open(csv_path, "w", newline="", encoding="utf-8") as fo:
                                writer = csv.writer(fo)
                                for row in table:
                                    writer.writerow([("" if cell is None else str(cell)).strip() for cell in row])
                            # also create text fallback
                            txt_path = csv_path + ".txt"
                            with open(txt_path, "w", encoding="utf-8") as fo:
                                for row in table:
                                    fo.write(" | ".join([("" if cell is None else str(cell)).strip() for cell in row]) + "\n")
                            out.append({
                                "page": i+1,
                                "csv_path": csv_path,
                                "txt_path": txt_path,
                                "rows": len(table)
                            })
                        except Exception as e:
                            logger.debug("Failed to write table csv: %s", e)
    except Exception as e:
        logger.warning("pdfplumber failed for %s: %s", pdf_path, e)
    return out

def extract_text_blocks_from_page(doc: fitz.Document, page_num: int) -> List[Dict[str, Any]]:
    """
    Use PyMuPDF page.get_text('dict') to get blocks with coordinates.
    Returns list of blocks: {type: 'text', text, bbox, block_no}
    """
    page = doc.load_page(page_num)
    textdict = page.get_text("dict")
    blocks = []
    for b_i, block in enumerate(textdict.get("blocks", [])):
        # block types: 0=text, 1=image
        if block.get("type") == 0:
            # build text
            lines = []
            for line in block.get("lines", []):
                spans = [span.get("text", "") for span in line.get("spans", [])]
                lines.append("".join(spans))
            text = "\n".join(lines).strip()
            if text:
                blocks.append({
                    "type": "text",
                    "text": text,
                    "bbox": block.get("bbox"),
                    "block_no": b_i
                })
    return blocks

# --- High-level per-book processing ---
def process_book(pdf_filename: str, textbook_title: str, output_root: str = OUTPUT_DIR):
    pdf_path = os.path.join(PDF_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)
    book_slug = safe_slug(textbook_title or pathlib.Path(pdf_filename).stem)
    out_base = os.path.join(output_root, book_slug)
    os.makedirs(out_base, exist_ok=True)
    assets_dir = os.path.join(out_base, ASSETS_SUBDIR)
    os.makedirs(assets_dir, exist_ok=True)

    logger.info("Processing book %s -> %s", pdf_filename, out_base)

    # 1) Extract tables (pdfplumber)
    logger.info("Extracting tables with pdfplumber (best-effort)...")
    tables_meta = extract_tables_from_pdf(pdf_path, assets_dir)
    logger.info("Found %d tables (saved to assets).", len(tables_meta))

    # 2) Open with PyMuPDF and extract pages (text blocks & images)
    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    logger.info("Opened PDF with %d pages", n_pages)

    page_tasks = []
    extracted_items = []  # collect items here: text blocks, image assets, tables
    # Use ThreadPoolExecutor to parallelize per-page extraction (bounded)
    with ThreadPoolExecutor(max_workers=MAX_PAGE_WORKERS) as exe:
        futures = {}
        for pnum in range(n_pages):
            futures[exe.submit(process_page_worker, pdf_path, pnum, assets_dir, book_slug)] = pnum
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Pages processed"):
            pnum = futures[fut]
            try:
                items = fut.result()
                # items is list of dicts
                extracted_items.extend(items)
            except Exception as e:
                logger.warning("Page %d extraction failed: %s", pnum+1, e)

    # Add table items from pdfplumber into extracted_items (map by page)
    for t in tables_meta:
        extracted_items.append({
            "type": "table_asset",
            "page": t["page"],
            "csv_path": t["csv_path"],
            "txt_path": t["txt_path"],
            "rows": t.get("rows", 0)
        })

    # 3) Convert extracted items into chunks
    logger.info("Converting extracted items into chunks...")
    chunks = []
    manifest = []
    chunk_counter = 0

    # Sort items by page for consistent chunk order
    extracted_items.sort(key=lambda x: (x.get("page", 0), x.get("type", "")))

    for it in extracted_items:
        it_type = it.get("type")
        page = it.get("page")
        if it_type == "text":
            text = it.get("text", "")
            for part in chunk_text(text):
                cid = f"{book_slug}__chunk_{chunk_counter:06d}"
                metadata = {
                    "textbook": textbook_title,
                    "source_file": pdf_filename,
                    "page": page,
                    "block_type": "text",
                    "block_no": it.get("block_no")
                }
                chunks.append({"id": cid, "text": part, "metadata": metadata})
                manifest.append({"id": cid, **metadata})
                chunk_counter += 1
        elif it_type == "image":
            # store as a short description + OCR text if present
            txt = it.get("ocr_text", "") or ""
            desc = f"[IMAGE on page {page}]"
            content = desc + ("\n\nOCR_TEXT:\n" + txt if txt else "")
            cid = f"{book_slug}__chunk_{chunk_counter:06d}"
            metadata = {
                "textbook": textbook_title,
                "source_file": pdf_filename,
                "page": page,
                "block_type": "image",
                "asset_path": it.get("asset_path"),
                "likely_formula": it.get("likely_formula", False)
            }
            chunks.append({"id": cid, "text": content, "metadata": metadata})
            manifest.append({"id": cid, **metadata})
            chunk_counter += 1
        elif it_type == "table_asset":
            # read txt_path fallback to include as chunk text
            txt_path = it.get("txt_path")
            try:
                with open(txt_path, "r", encoding="utf-8") as fo:
                    ttext = fo.read()
            except Exception:
                ttext = ""
            cid = f"{book_slug}__chunk_{chunk_counter:06d}"
            metadata = {
                "textbook": textbook_title,
                "source_file": pdf_filename,
                "page": page,
                "block_type": "table",
                "csv_path": it.get("csv_path")
            }
            chunks.append({"id": cid, "text": ("[TABLE]\n" + ttext).strip(), "metadata": metadata})
            manifest.append({"id": cid, **metadata})
            chunk_counter += 1
        else:
            # unknown -> stringify
            cid = f"{book_slug}__chunk_{chunk_counter:06d}"
            metadata = {"textbook": textbook_title, "source_file": pdf_filename, "page": page, "block_type": it_type}
            chunks.append({"id": cid, "text": str(it.get("text", ""))[:2000], "metadata": metadata})
            manifest.append({"id": cid, **metadata})
            chunk_counter += 1

    # 4) Save outputs
    chunks_path = os.path.join(out_base, "chunks.jsonl")
    manifest_path = os.path.join(out_base, "manifest.csv")
    with open(chunks_path, "w", encoding="utf-8") as fo:
        for c in chunks:
            fo.write(json.dumps(c, ensure_ascii=False) + "\n")
    # manifest CSV
    if manifest:
        keys = list(manifest[0].keys())
        with open(manifest_path, "w", newline="", encoding="utf-8") as fo:
            writer = csv.DictWriter(fo, fieldnames=keys)
            writer.writeheader()
            writer.writerows(manifest)

    logger.info("Saved %d chunks to %s and manifest to %s", len(chunks), chunks_path, manifest_path)
    logger.info("Assets saved under %s", assets_dir)

    # OPTIONAL: Embeddings + upsert to Qdrant (uncomment and configure)
    # ollama = OllamaEmbeddings(model="qwen3-embedding:4b")
    # texts = [c["text"] for c in chunks]
    # embeddings = ollama.embed_documents(texts)
    # q = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
    # # create collection if needed
    # if not q.get_collections().collections:
    #     q.recreate_collection(collection_name="books", vectors_config={"size": len(embeddings[0]), "distance": "Cosine"})
    # points = []
    # for idx, emb in enumerate(embeddings):
    #     meta = chunks[idx]["metadata"]
    #     points.append(PointStruct(id=int(idx), vector=emb, payload=meta))
    # q.upsert(collection_name="books", points=points)
    # logger.info("Upserted embeddings to Qdrant collection 'books'")

def process_page_worker(pdf_path: str, page_num: int, assets_dir: str, book_slug: str) -> List[Dict[str, Any]]:
    """
    Worker to extract text blocks and images for a single page.
    Returns list of item dicts.
    """
    items = []
    try:
        # open doc in worker (fitz doc is not thread-safe across threads)
        doc = fitz.open(pdf_path)
        # text blocks
        try:
            text_blocks = extract_text_blocks_from_page(doc, page_num)
            for b in text_blocks:
                items.append({
                    "type": "text",
                    "page": page_num + 1,
                    "text": b["text"],
                    "block_no": b.get("block_no"),
                    "bbox": b.get("bbox")
                })
        except Exception as e:
            logger.debug("Text blocks extraction failed page %d: %s", page_num+1, e)

        # images
        try:
            images = extract_images_from_page(doc, page_num, assets_dir, book_slug)
            for im in images:
                items.append({
                    "type": "image",
                    "page": im["page"],
                    "asset_path": im["asset_path"],
                    "ocr_text": im.get("ocr_text", ""),
                    "likely_formula": im.get("likely_formula", False),
                    "width": im.get("width"),
                    "height": im.get("height")
                })
        except Exception as e:
            logger.debug("Image extraction failed page %d: %s", page_num+1, e)

        doc.close()
    except Exception as e:
        logger.warning("process_page_worker error for page %d: %s", page_num+1, e)
    return items

# --- main runner ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdfs = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    if not pdfs:
        logger.error("No PDFs in %s", PDF_DIR)
        return
    logger.info("Found %d PDFs", len(pdfs))
    for pdf in pdfs:
        title = TEXTBOOK_MAP.get(pdf, pathlib.Path(pdf).stem)
        try:
            process_book(pdf, title)
        except Exception as e:
            logger.exception("Failed processing %s: %s", pdf, e)

if __name__ == "__main__":
    main()
