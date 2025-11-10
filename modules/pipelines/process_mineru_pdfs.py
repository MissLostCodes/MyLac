"""
Pipeline to upload local PDFs to Mineru, poll results,
download parse outputs, convert to LLM-friendly chunks with textbook metadata,
and save JSONL + manifest CSV for indexing.
"""

import os
import sys
import time
import json
import csv
import zipfile
import logging
from typing import List, Dict, Any

import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# === CONFIG ===
MINERU_BASE = "https://mineru.net/api/v4"
UPLOAD_URL_BATCH = f"{MINERU_BASE}/file-urls/batch"
BATCH_RESULT_URL = f"{MINERU_BASE}/extract-results/batch"
SINGLE_TASK_URL = f"{MINERU_BASE}/extract/task"
SINGLE_TASK_QUERY = f"{MINERU_BASE}/extract/task"  # GET /{task_id}

PDF_DIR = "../../KnowledgeBase"
OUTPUT_DIR = "../outputs"
MAX_CONCURRENT_UPLOADS = 3
REQUEST_TIMEOUT = 60
POLL_INTERVAL = 10
MAX_POLL_WAIT = 60 * 60 * 2
RETRY_TRIES = 5
RETRY_BACKOFF = 2

TARGET_TOKENS = 700
TOKEN_CHAR_ESTIMATE = 4
OVERLAP_TOKENS = 75

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mineru_pipeline")


def env_token() -> str:
    tok = os.getenv("MINERU_TOKEN")
    if not tok:
        logger.error("Please set MINERU_TOKEN environment variable.")
        sys.exit(1)
    return tok


def list_pdfs() -> List[str]:
    files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    if not files:
        logger.error("No PDFs found in %s", PDF_DIR)
        sys.exit(1)
    return files


def request_with_retry(method, url, **kwargs):
    tries = 0
    backoff = 1
    while tries < RETRY_TRIES:
        try:
            resp = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            tries += 1
            logger.warning("Request error %s %s (try %d/%d): %s", method, url, tries, RETRY_TRIES, e)
            if tries >= RETRY_TRIES:
                raise
            time.sleep(backoff)
            backoff *= RETRY_BACKOFF


def request_upload_urls(files: List[Dict[str, Any]], token: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "enable_formula": True,
        "language": "en",
        "enable_table": True,
        "files": files
    }
    resp = request_with_retry("POST", UPLOAD_URL_BATCH, headers=headers, json=payload)
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"Mineru returned error: {data}")
    return data["data"]


def upload_file_to_presigned(url: str, local_path: str, max_retries: int = 3):
    file_size = os.path.getsize(local_path)
    print(f"Uploading {local_path} ({file_size / (1024 * 1024):.2f} MB)...")

    with open(local_path, "rb") as f:
        data = f.read()

    for attempt in range(1, max_retries + 1):
        try:
            with tqdm(total=file_size, unit="B", unit_scale=True, desc=f"Uploading (try {attempt})", ncols=100) as pbar:
                chunk_size = 1024 * 1024  # 1 MB
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i + chunk_size]
                    resp = requests.put(url, data=chunk, timeout=(30, 10000))
                    resp.raise_for_status()
                    pbar.update(len(chunk))
            print("✅ Upload completed successfully!")
            return
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Upload error on attempt {attempt}: {e}")
            if attempt < max_retries:
                print("Retrying with a fresh presigned URL...")
                continue
            else:
                raise


def safe_upload(file_obj, local_path, token: str, max_attempts: int = 3):
    for attempt in range(max_attempts):
        try:
            urls_resp = request_upload_urls([file_obj], token)
            upload_url = urls_resp["file_urls"][0]
            upload_file_to_presigned(upload_url, local_path)
            return urls_resp
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logger.warning("⚠️ Presigned URL expired. Retrying attempt %d/%d...", attempt + 1, max_attempts)
                continue
            raise
    raise RuntimeError(f"Failed to upload {local_path} after {max_attempts} attempts")


def poll_batch_result(batch_id: str, token: str, timeout_seconds: int = MAX_POLL_WAIT) -> Dict[str, Any]:
    url = f"{BATCH_RESULT_URL}/{batch_id}"
    headers = {"Authorization": f"Bearer {token}"}
    start = time.time()
    interval = POLL_INTERVAL
    while True:
        resp = request_with_retry("GET", url, headers=headers)
        data = resp.json()
        extract_result = data.get("data", {}).get("extract_result")
        if extract_result:
            if all(r.get("state") in ("done", "failed") for r in extract_result):
                return data["data"]
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Polling exceeded {timeout_seconds} seconds for batch {batch_id}")
        logger.info("Batch %s not done yet. Sleeping %ds", batch_id, interval)
        time.sleep(interval)
        interval = min(interval * 1.5, 300)


def download_and_extract(zip_url: str, dest_dir: str, token: str = None):
    os.makedirs(dest_dir, exist_ok=True)
    logger.info("Downloading %s -> %s", zip_url, dest_dir)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = request_with_retry("GET", zip_url, headers=headers, stream=True)
    local_zip = os.path.join(dest_dir, "mineru_result.zip")
    with open(local_zip, "wb") as fo:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fo.write(chunk)
    logger.info("Downloaded zip to %s, extracting...", local_zip)
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(dest_dir)
    logger.info("Extracted to %s", dest_dir)
    return dest_dir


def safe_slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)[:120]


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / TOKEN_CHAR_ESTIMATE))


def split_text_to_chunks(text: str, target_tokens: int = TARGET_TOKENS, overlap_tokens: int = OVERLAP_TOKENS):
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
            chunks.append(cur)
            overlap_chars = int(overlap_tokens * TOKEN_CHAR_ESTIMATE)
            cur = (cur[-overlap_chars:] + "\n\n" + p).strip()
            cur_tokens = estimate_tokens(cur)
    if cur.strip():
        chunks.append(cur)
    return chunks


def load_mineru_jsons(extracted_dir: str) -> List[Dict[str, Any]]:
    results = []
    for root, _, files in os.walk(extracted_dir):
        for fn in files:
            path = os.path.join(root, fn)
            if fn.endswith(".json"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    results.append({"path": path, "type": "json", "data": data})
                except Exception as e:
                    logger.warning("Failed to load json %s: %s", path, e)
            elif fn.endswith((".md", ".markdown", ".html")):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                results.append({"path": path, "type": "raw", "data": raw})
    return results


def parse_mineru_structured_json(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []

    def walk(obj):
        if isinstance(obj, dict):
            t = obj.get("type")
            if t in ("table", "Table"):
                out.append({"type": "table", "page": obj.get("page"), "content": obj})
                return
            if t in ("formula", "Formula"):
                out.append({"type": "formula", "page": obj.get("page"), "content": obj})
                return
            if "text" in obj and isinstance(obj["text"], str):
                out.append({"type": "text", "page": obj.get("page"), "content": obj["text"], "raw": obj})
                return
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for e in obj:
                walk(e)

    walk(js)
    return out


def write_jsonl(rows: List[Dict[str, Any]], out_path: str):
    with open(out_path, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_manifest(manifest_rows: List[Dict[str, Any]], csv_path: str):
    if not manifest_rows:
        return
    keys = list(manifest_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, fieldnames=keys)
        writer.writeheader()
        writer.writerows(manifest_rows)

#

def process_book(local_filename: str, textbook_title: str, token: str):
    book_slug = safe_slug(textbook_title)
    out_base = os.path.join(OUTPUT_DIR, book_slug)
    os.makedirs(out_base, exist_ok=True)

    local_path = os.path.join(PDF_DIR, local_filename)
    logger.info("Uploading %s -> Mineru", local_path)

    file_obj = {"name": local_filename, "is_ocr": True, "data_id": safe_slug(textbook_title)}
    urls_resp = safe_upload(file_obj, local_path, token)
    logger.info("Upload complete. Mineru will auto-submit parsing task.")

    # --- NEW FIX ---
    batch_id = urls_resp.get("batch_id")
    if not batch_id:
        # Single task
        upload_url = urls_resp["file_urls"][0]
        payload = {
            "url": upload_url,
            "is_ocr": True,
            "enable_formula": True,
            "enable_table": True,
            "data_id": file_obj["data_id"]
        }
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        resp = request_with_retry("POST", SINGLE_TASK_URL, headers=headers, json=payload)
        task_id = resp.json()["data"]["task_id"]
    else:
        # Batch task
        task_id = batch_id

    # Poll until done
    poll_url = f"{SINGLE_TASK_QUERY}/{task_id}"
    start = time.time()
    while True:
        r = request_with_retry("GET", poll_url, headers=headers)
        data = r.json().get("data", {})
        state = data.get("state")
        if state == "done":
            full_zip = data.get("full_zip_url")
            if not full_zip:
                raise RuntimeError(f"No full_zip_url returned for {local_filename}")
            extract_dir = os.path.join(out_base, "mineru_raw")
            download_and_extract(full_zip, extract_dir)
            break
        elif state == "failed":
            raise RuntimeError(f"Mineru parsing failed for {local_filename}")
        elif time.time() - start > MAX_POLL_WAIT:
            raise TimeoutError(f"Polling timeout for {local_filename}")
        logger.info("Task state=%s for %s. Sleeping...", state, local_filename)
        time.sleep(POLL_INTERVAL)

def main():
    token = env_token()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdfs = list_pdfs()
    logger.info("Found %d PDFs", len(pdfs))

    for pdf in pdfs:
        title = os.path.splitext(pdf)[0]  # use filename as title
        logger.info("Processing %s (%s)", pdf, title)
        try:
            process_book(pdf, title, token)
        except Exception as e:
            logger.error("❌ Error processing %s: %s", pdf, e)
    logger.info("🎉 All books processed successfully!")


if __name__ == "__main__":
    main()
