# # """
# # Pipeline to upload local PDFs to Mineru, poll results,
# # download parse outputs, convert to LLM-friendly chunks with textbook metadata,
# # and save JSONL + manifest CSV for indexing.
# # """
# #
# # import os
# # import sys
# # import time
# # import json
# # import csv
# # import shutil
# # import zipfile
# # import hashlib
# # import logging
# # import pathlib
# # import threading
# # from typing import List, Dict, Any
# # from concurrent.futures import ThreadPoolExecutor, as_completed
# #
# # import requests
# # from tqdm import tqdm
# # from dateutil import parser as dateparser
# # from dotenv import load_dotenv
# #
# # load_dotenv()
# #
# # logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# # logger = logging.getLogger("mineru_pipeline")
# #
# # # === CONFIG ===
# # MINERU_BASE = "https://mineru.net/api/v4"
# # UPLOAD_URL_BATCH = f"{MINERU_BASE}/file-urls/batch"
# # BATCH_RESULT_URL = f"{MINERU_BASE}/extract-results/batch"
# # SINGLE_TASK_URL = f"{MINERU_BASE}/extract/task"
# # SINGLE_TASK_QUERY = f"{MINERU_BASE}/extract/task"  # GET /{task_id}
# #
# #
# # PDF_DIR = "../KnowledgeBase"
# # OUTPUT_DIR = "./outputs"
# # MAX_CONCURRENT_UPLOADS = 3
# # REQUEST_TIMEOUT = 60
# # POLL_INTERVAL = 10                # seconds between status checks (backoff applies)
# # MAX_POLL_WAIT = 60 * 60 * 2       # 2 hours max wait
# # RETRY_TRIES = 5
# # RETRY_BACKOFF = 2                 # multiplier
# #
# # # Chunking parameters
# # TARGET_TOKENS = 700
# # TOKEN_CHAR_ESTIMATE = 4           # heuristic: 1 token : 4 chars
# # OVERLAP_TOKENS = 75
# #
# # # Mapping for textbook metadata
# # TEXTBOOK_MAP = {
# #     "applied_ml_gopal.pdf": "Applied Machine Learning by Dr. M. Gopal",
# #     "intro_ml_alpaydin.pdf": "Introduction to Machine Learning by Ethem Alpaydın",
# #     "machine_learning_mitchell.pdf": "Machine Learning by Tom M. Mitchell"
# # }
# #
# # def env_token() -> str:
# #     tok = os.getenv("MINERU_TOKEN")
# #     if not tok:
# #         logger.error("Please set MINERU_TOKEN environment variable.")
# #         sys.exit(1)
# #     return tok
# #
# # def list_pdfs() -> List[str]:
# #     files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
# #     if not files:
# #         logger.error("No pdfs found in %s", PDF_DIR)
# #         sys.exit(1)
# #     return files
# #
# # # --- Network helpers with retries ---
# # def request_with_retry(method, url, **kwargs):
# #     tries = 0
# #     backoff = 1
# #     while tries < RETRY_TRIES:
# #         try:
# #             resp = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
# #             resp.raise_for_status()
# #             return resp
# #         except Exception as e:
# #             tries += 1
# #             logger.warning("Request error %s %s (try %d/%d): %s", method, url, tries, RETRY_TRIES, e)
# #             if tries >= RETRY_TRIES:
# #                 raise
# #             time.sleep(backoff)
# #             backoff *= RETRY_BACKOFF
# #
# # # --- Mineru integration  ---
# # def request_upload_urls(files: List[Dict[str, Any]], token: str) -> Dict[str, Any]:
# #     """
# #     Call Mineru /file-urls/batch to get upload URLs.
# #     files: list of {"name": "<filename>", "is_ocr": True, "data_id": "...", "page_ranges": "1-600"} objects
# #     """
# #     url = UPLOAD_URL_BATCH
# #     headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
# #     payload = {
# #         "enable_formula": True,
# #         "language": "en",
# #         "enable_table": True,
# #         "files": files
# #     }
# #     resp = request_with_retry("POST", url, headers=headers, json=payload)
# #     data = resp.json()
# #     if data.get("code") != 0:
# #         raise RuntimeError(f"Mineru returned error: {data}")
# #     return data["data"]
# #
# # # def upload_file_to_presigned(url: str, local_path: str):
# # #     with open(local_path, "rb") as f:
# # #         print("upload_file_to_presigned")
# # #         # Mineru no need to set Content-Type on upload
# # #         resp = request_with_retry("PUT", url, data=f)
# # #         print(resp.text)
# # #         if resp.status_code not in (200, 201):
# # #             raise RuntimeError(f"Upload failed status {resp.status_code} for {local_path}")
# #
# # from tqdm import tqdm
# # from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
# # import os
# #
# #
# # def upload_file_to_presigned(url: str, local_path: str):
# #     file_size = os.path.getsize(local_path)
# #     print(f"Uploading {local_path} ({file_size / (1024 * 1024):.2f} MB)...")
# #
# #     with open(local_path, "rb") as f:
# #         # Initialize tqdm progress bar
# #         with tqdm(total=file_size, unit='B', unit_scale=True, desc="Uploading", ncols=100) as pbar:
# #             def progress_monitor(monitor):
# #                 pbar.update(monitor.bytes_read - pbar.n)
# #
# #             # Wrap file with MultipartEncoder and Monitor
# #             encoder = MultipartEncoder(fields={"file": (os.path.basename(local_path), f, "application/pdf")})
# #             monitor = MultipartEncoderMonitor(encoder, progress_monitor)
# #
# #             headers = {"Content-Type": monitor.content_type}
# #
# #             resp = request_with_retry("PUT", url, data=monitor, headers=headers)
# #
# #     print(f"\nResponse code: {resp.status_code}")
# #     print("Response text:", resp.text[:500])  # Print first 500 chars to avoid long text
# #
# #     if resp.status_code not in (200, 201):
# #         raise RuntimeError(f"Upload failed status {resp.status_code} for {local_path}")
# #     else:
# #         print("✅ Upload completed successfully!")
# #
# #
# #
# # def safe_upload(file_obj, local_path, token: str, max_attempts: int = 3):
# #     """
# #     Attempts to upload file using fresh presigned URLs.
# #     Retries up to max_attempts if URL expires (HTTP 403).
# #     Returns the urls_resp for further processing (batch_id etc.)
# #     """
# #     for attempt in range(max_attempts):
# #         try:
# #             urls_resp = request_upload_urls([file_obj], token)
# #             upload_url = urls_resp["file_urls"][0]
# #             upload_file_to_presigned(upload_url, local_path)
# #             return urls_resp
# #         except requests.exceptions.HTTPError as e:
# #             if e.response.status_code == 403:
# #                 logger.warning("⚠️ Presigned URL expired. Retrying attempt %d/%d...", attempt+1, max_attempts)
# #                 continue
# #             raise
# #     raise RuntimeError(f"Failed to upload {local_path} after {max_attempts} attempts")
# #
# # def poll_batch_result(batch_id: str, token: str, timeout_seconds: int = MAX_POLL_WAIT) -> Dict[str, Any]:
# #     url = f"{BATCH_RESULT_URL}/{batch_id}"
# #     headers = {"Authorization": f"Bearer {token}"}
# #     start = time.time()
# #     interval = POLL_INTERVAL
# #     while True:
# #         resp = request_with_retry("GET", url, headers=headers)
# #         data = resp.json()
# #         if data.get("code") != 0:
# #             logger.warning("non-zero code polling batch: %s", data)
# #         extract_result = data.get("data", {}).get("extract_result")
# #         state_ok = True
# #         all_done = True
# #         if isinstance(extract_result, list):
# #             for r in extract_result:
# #                 state = r.get("state")
# #                 if state in ("failed",):
# #                     logger.error("One file failed parsing: %s", r.get("file_name"))
# #                     # we continue but will include failure in result
# #                 if state not in ("done", "failed"):
# #                     all_done = False
# #                     state_ok = False
# #         else:
# #             # single file result
# #             s = data.get("data", {}).get("state")
# #             if s not in ("done", "failed"):
# #                 all_done = False
# #                 state_ok = False
# #
# #         if all_done:
# #             return data["data"]
# #         if time.time() - start > timeout_seconds:
# #             raise TimeoutError(f"Polling exceeded {timeout_seconds} seconds for batch {batch_id}")
# #         logger.info("Batch %s not done yet. sleeping %ds", batch_id, interval)
# #         time.sleep(interval)
# #         interval = min(interval * 1.5, 300)  # backoff to max 5 minutes
# #
# # def download_and_extract(zip_url: str, dest_dir: str, token: str = None):
# #     os.makedirs(dest_dir, exist_ok=True)
# #     logger.info("Downloading %s -> %s", zip_url, dest_dir)
# #     headers = {}
# #     if token: headers["Authorization"] = f"Bearer {token}"
# #     resp = request_with_retry("GET", zip_url, headers=headers, stream=True)
# #     local_zip = os.path.join(dest_dir, "mineru_result.zip")
# #     with open(local_zip, "wb") as fo:
# #         for chunk in resp.iter_content(chunk_size=1024*1024):
# #             if chunk:
# #                 fo.write(chunk)
# #     logger.info("Downloaded zip to %s, extracting...", local_zip)
# #     with zipfile.ZipFile(local_zip, 'r') as zf:
# #         zf.extractall(dest_dir)
# #     logger.info("Extracted to %s", dest_dir)
# #     return dest_dir
# #
# # # ---- Parsing mineru outputs into chunks ----
# # def safe_slug(s: str) -> str:
# #     return "".join(c if c.isalnum() else "_" for c in s)[:120]
# #
# # def estimate_tokens(text: str) -> int:
# #     # heuristic: 1 token ≈ TOKEN_CHAR_ESTIMATE characters
# #     return max(1, int(len(text) / TOKEN_CHAR_ESTIMATE))
# #
# # def split_text_to_chunks(text: str, target_tokens: int = TARGET_TOKENS, overlap_tokens: int = OVERLAP_TOKENS):
# #     """
# #     Simple split on paragraphs / sentences into ~target_tokens using char heuristic.
# #     Returns list of chunks (strings).
# #     """
# #     # normalize
# #     paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
# #     chunks = []
# #     cur = ""
# #     cur_tokens = 0
# #     for p in paragraphs:
# #         p_tokens = estimate_tokens(p)
# #         if cur_tokens + p_tokens <= target_tokens or cur_tokens == 0:
# #             if cur:
# #                 cur += "\n\n" + p
# #             else:
# #                 cur = p
# #             cur_tokens += p_tokens
# #         else:
# #             chunks.append(cur)
# #             # build next chunk with overlap
# #             # approximate overlap by taking last X chars
# #             overlap_chars = int(overlap_tokens * TOKEN_CHAR_ESTIMATE)
# #             cur = (cur[-overlap_chars:] + "\n\n" + p).strip()
# #             cur_tokens = estimate_tokens(cur)
# #     if cur.strip():
# #         chunks.append(cur)
# #     return chunks
# #
# # def load_mineru_jsons(extracted_dir: str) -> List[Dict[str, Any]]:
# #     """
# #     Mineru output may include JSON or markdown or html files.
# #     We'll walk extracted_dir and find .json (structured) first; if none, parse md/html.
# #     Returns list of dicts: {'path':..., 'content':parsed_json or raw text}
# #     """
# #     results = []
# #     for root, _, files in os.walk(extracted_dir):
# #         for fn in files:
# #             lfn = fn.lower()
# #             path = os.path.join(root, fn)
# #             if lfn.endswith(".json"):
# #                 try:
# #                     with open(path, "r", encoding="utf-8") as f:
# #                         data = json.load(f)
# #                     results.append({"path": path, "type": "json", "data": data})
# #                 except Exception as e:
# #                     logger.warning("Failed to load json %s: %s", path, e)
# #             elif lfn.endswith(".md") or lfn.endswith(".markdown") or lfn.endswith(".html"):
# #                 with open(path, "r", encoding="utf-8", errors="ignore") as f:
# #                     raw = f.read()
# #                 results.append({"path": path, "type": "raw", "data": raw})
# #             # images and other assets will be handled later by scanning assets dir
# #     return results
# #
# # def parse_mineru_structured_json(js: Dict[str, Any]) -> List[Dict[str, Any]]:
# #     """
# #     Mineru structured JSON varies by model version; attempt to locate text blocks, table blocks, images, formulas.
# #     We'll produce a list of items like:
# #       {'type': 'text'|'table'|'image'|'formula', 'page': int, 'content': str or table-structure, 'raw': original}
# #     This parser is intentionally flexible and best-effort.
# #     """
# #     out = []
# #     # Common keys: might have 'pages' or 'blocks' or 'content'
# #     def walk(obj):
# #         if isinstance(obj, dict):
# #             # heuristics: keys like 'type' == 'table' or 'text'
# #             t = obj.get("type")
# #             if t in ("table", "Table"):
# #                 out.append({"type": "table", "page": obj.get("page", None), "content": obj})
# #                 return
# #             if t in ("formula", "Formula"):
# #                 out.append({"type": "formula", "page": obj.get("page", None), "content": obj})
# #                 return
# #             if "text" in obj and isinstance(obj.get("text"), str) and len(obj.get("text").strip())>0:
# #                 out.append({"type": "text", "page": obj.get("page", obj.get("page_no", None)), "content": obj.get("text"), "raw": obj})
# #                 return
# #             # specific mineru structures: maybe 'pages' list
# #             for k, v in obj.items():
# #                 walk(v)
# #         elif isinstance(obj, list):
# #             for e in obj:
# #                 walk(e)
# #     walk(js)
# #     return out
# #
# # def write_jsonl(rows: List[Dict[str, Any]], out_path: str):
# #     with open(out_path, "w", encoding="utf-8") as fo:
# #         for r in rows:
# #             fo.write(json.dumps(r, ensure_ascii=False) + "\n")
# #
# # def save_manifest(manifest_rows: List[Dict[str, Any]], csv_path: str):
# #     if not manifest_rows:
# #         return
# #     keys = list(manifest_rows[0].keys())
# #     with open(csv_path, "w", newline="", encoding="utf-8") as fo:
# #         writer = csv.DictWriter(fo, fieldnames=keys)
# #         writer.writeheader()
# #         for r in manifest_rows:
# #             writer.writerow(r)
# #
# # # Main pipeline per book
# # def process_book(local_filename: str, textbook_title: str, token: str):
# #     book_slug = safe_slug(textbook_title or local_filename)
# #     out_base = os.path.join(OUTPUT_DIR, book_slug)
# #     os.makedirs(out_base, exist_ok=True)
# #     logf = os.path.join(out_base, "log.txt")
# #     fh = logging.FileHandler(logf)
# #     fh.setLevel(logging.INFO)
# #     logger.addHandler(fh)
# #     #
# #     # # 1) request upload URL
# #     # # file_obj = {"name": local_filename, "is_ocr": True, "data_id": textbook_title.replace(" ", "_")}
# #     # # urls_resp = request_upload_urls([file_obj], token)
# #     # # # Mineru returns file_urls: list
# #     # # file_urls = urls_resp.get("file_urls") or urls_resp.get("file_urls") or []
# #     # # if not file_urls:
# #     # #     # Fallback: maybe single file return format
# #     # #     file_urls = urls_resp.get("file_urls", [])
# #     # # # Usually file_urls is list-of-urls with same order
# #     # # if isinstance(file_urls, list) and len(file_urls) >= 1:
# #     # #     upload_url = file_urls[0]
# #     # # else:
# #     # #     raise RuntimeError("No upload url returned from mineru")
# #     # #
# #     # # local_path = os.path.join(PDF_DIR, local_filename)
# #     # # logger.info("Uploading %s -> mineru", local_path)
# #     # # upload_file_to_presigned(upload_url, local_path)
# #     # # logger.info("Upload complete. Mineru will auto-submit parsing task.")
# #     # local_path = os.path.join(PDF_DIR, local_filename)
# #     # logger.info("Uploading %s -> mineru", local_path)
# #     #
# #     # file_obj = {"name": local_filename, "is_ocr": True, "data_id": textbook_title.replace(" ", "_")}
# #     # urls_resp = safe_upload(file_obj, local_path, token)
# #     # logger.info("Upload complete. Mineru will auto-submit parsing task.")
# #     # # If batch flow returns batch_id, we should get it. The initial request returns batch_id when multiple files requested.
# #     # batch_id = urls_resp.get("batch_id") or urls_resp.get("batch_id")
# #     # if not batch_id:
# #     #     # If no batch_id, Mineru maybe submitted single task, we need to request single task by calling /extract/task with URL
# #     #     # Simpler: call Single File Parsing with url param to create a task pointing to the file url (if supported)
# #     #     # We'll attempt the single-file create using SINGLE_TASK_URL to be robust.
# #     #     # Build request: {"url": upload_url, "is_ocr": True, "enable_formula": True, "enable_table": True, "data_id": file_obj["data_id"]}
# #     #     payload = {"url": upload_url, "is_ocr": True, "enable_formula": True, "enable_table": True, "data_id": file_obj["data_id"]}
# #     #     headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
# #     #     resp = request_with_retry("POST", SINGLE_TASK_URL, headers=headers, json=payload)
# #     #     d = resp.json()
# #     #     if d.get("code") != 0:
# #     #         logger.error("Failed to create single task: %s", d)
# #     #         raise RuntimeError("Failed create single task")
# #     #     batch_id = d["data"]["task_id"]  # use task_id as batch for polling below
# #     #     # Note: We will poll SINGLE_TASK_QUERY/{task_id} instead of batch endpoint
# #     #     poll_url = f"{SINGLE_TASK_QUERY}/{batch_id}"
# #     #     # Poll loop:
# #     #     start = time.time()
# #     #     while True:
# #     #         r = request_with_retry("GET", poll_url, headers=headers)
# #     #         j = r.json()
# #     #         state = j.get("data", {}).get("state")
# #     #         if state == "done":
# #     #             full_zip = j["data"].get("full_zip_url")
# #     #             if not full_zip:
# #     #                 raise RuntimeError("No full_zip_url returned for single task")
# #     #             extract_dir = os.path.join(out_base, "mineru_raw")
# #     #             download_and_extract(full_zip, extract_dir, token=None)
# #     #             break
# #     #         if state == "failed":
# #     #             raise RuntimeError("Parsing failed for task: %s" % j)
# #     #         if time.time() - start > MAX_POLL_WAIT:
# #     #             raise TimeoutError("Polling timeout for single task")
# #     #         logger.info("Single task %s state=%s. Sleeping...", batch_id, state)
# #     #         time.sleep(POLL_INTERVAL)
# #     # else:
# #     #     # Poll the batch
# #     #     logger.info("Batch submitted: %s. Polling for completion...", batch_id)
# #     #     batch_data = poll_batch_result(batch_id, token)
# #     #     # batch_data.extract_result contains list with full_zip_url per file
# #     #     extract_results = batch_data.get("extract_result") or batch_data.get("files") or []
# #     #     # find our file entry
# #     #     chosen = None
# #     #     if isinstance(extract_results, list):
# #     #         for r in extract_results:
# #     #             if r.get("file_name") == local_filename or r.get("file_name", "").endswith(local_filename):
# #     #                 chosen = r
# #     #                 break
# #     #         if not chosen:
# #     #             chosen = extract_results[0]  # fallback
# #     #     else:
# #     #         chosen = extract_results
# #     #     full_zip = chosen.get("full_zip_url") or batch_data.get("full_zip_url")
# #     #     if not full_zip:
# #     #         raise RuntimeError("No full_zip_url for parsed file")
# #     #     extract_dir = os.path.join(out_base, "mineru_raw")
# #     #     download_and_extract(full_zip, extract_dir, token=None)
# #     # book_slug = safe_slug(textbook_title or local_filename)
# #     # out_base = os.path.join(OUTPUT_DIR, book_slug)
# #     # os.makedirs(out_base, exist_ok=True)
# #
# #     local_path = os.path.join(PDF_DIR, local_filename)
# #     logger.info("Uploading %s -> Mineru", local_path)
# #
# #     # Prepare file object
# #     file_obj = {"name": local_filename, "is_ocr": True, "data_id": textbook_title.replace(" ", "_")}
# #
# #     # Use safe_upload to mimic your working snippet
# #     urls_resp = safe_upload(file_obj, local_path, token)
# #     logger.info("Upload complete. Mineru will auto-submit parsing task.")
# #
# #     # Fetch batch_id
# #     batch_id = urls_resp.get("batch_id")
# #     if not batch_id:
# #         # fallback: create single task if batch_id missing
# #         upload_url = urls_resp["file_urls"][0]
# #         payload = {"url": upload_url, "is_ocr": True, "enable_formula": True, "enable_table": True, "data_id": file_obj["data_id"]}
# #         headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
# #         resp = request_with_retry("POST", SINGLE_TASK_URL, headers=headers, json=payload)
# #         d = resp.json()
# #         if d.get("code") != 0:
# #             raise RuntimeError("Failed to create single task: %s" % d)
# #         batch_id = d["data"]["task_id"]
# #         poll_url = f"{SINGLE_TASK_QUERY}/{batch_id}"
# #         # poll single task
# #         start = time.time()
# #         while True:
# #             r = request_with_retry("GET", poll_url, headers=headers)
# #             state = r.json().get("data", {}).get("state")
# #             if state == "done":
# #                 full_zip = r.json()["data"]["full_zip_url"]
# #                 extract_dir = os.path.join(out_base, "mineru_raw")
# #                 download_and_extract(full_zip, extract_dir)
# #                 break
# #             elif state == "failed":
# #                 raise RuntimeError("Parsing failed for task")
# #             elif time.time() - start > MAX_POLL_WAIT:
# #                 raise TimeoutError("Polling timeout")
# #             logger.info("Single task state=%s. Sleeping...", state)
# #             time.sleep(POLL_INTERVAL)
# #     else:
# #         # batch polling
# #         logger.info("Batch submitted: %s. Polling for completion...", batch_id)
# #         batch_data = poll_batch_result(batch_id, token)
# #         extract_results = batch_data.get("extract_result") or []
# #         chosen = next((r for r in extract_results if r.get("file_name") == local_filename), extract_results[0])
# #         full_zip = chosen.get("full_zip_url") or batch_data.get("full_zip_url")
# #         if not full_zip:
# #             raise RuntimeError("No full_zip_url for parsed file")
# #         extract_dir = os.path.join(out_base, "mineru_raw")
# #         download_and_extract(full_zip, extract_dir)
# #
# #     # Proceed with parsing and chunking as before
# #     parsed_files = load_mineru_jsons(extract_dir)
# #     items = []
# #     for p in parsed_files:
# #         if p["type"] == "json":
# #             items.extend(parse_mineru_structured_json(p["data"]))
# #         else:
# #             items.append({"type": "text", "page": None, "content": p["data"], "raw_path": p["path"]})
# #     # Now parse mineru outputs
# #     parsed_files = load_mineru_jsons(extract_dir)
# #     items = []
# #     for p in parsed_files:
# #         if p["type"] == "json":
# #             items.extend(parse_mineru_structured_json(p["data"]))
# #         else:
# #             # raw markdown/html -> treat as big text block
# #             items.append({"type": "text", "page": None, "content": p["data"], "raw_path": p["path"]})
# #
# #     # Also collect assets (images/tables saved as files)
# #     assets_dir = os.path.join(out_base, "assets")
# #     os.makedirs(assets_dir, exist_ok=True)
# #     # copy all non-json/md/html extracted files into assets
# #     for root, _, files in os.walk(extract_dir):
# #         for fn in files:
# #             if not fn.lower().endswith((".json", ".md", ".html", ".markdown")):
# #                 src = os.path.join(root, fn)
# #                 dst = os.path.join(assets_dir, fn)
# #                 try:
# #                     shutil.copy2(src, dst)
# #                 except Exception:
# #                     pass
# #
# #     # Convert items into chunks
# #     chunk_rows = []
# #     manifest_rows = []
# #     chunk_counter = 0
# #     for it in items:
# #         it_type = it.get("type", "text")
# #         page = it.get("page")
# #         if it_type == "text":
# #             text = it.get("content", "")
# #             chunks = split_text_to_chunks(text)
# #             for c in chunks:
# #                 chunk_id = f"{book_slug}__chunk_{chunk_counter:06d}"
# #                 row = {
# #                     "id": chunk_id,
# #                     "textbook": textbook_title,
# #                     "source_file": local_filename,
# #                     "page": page,
# #                     "type": "text",
# #                     "content": c.strip()
# #                 }
# #                 chunk_rows.append(row)
# #                 manifest_rows.append({"id": chunk_id, "textbook": textbook_title, "page": page or "", "type": "text"})
# #                 chunk_counter += 1
# #         elif it_type == "table":
# #             # try to serialize table structure or raw html/text inside content
# #             chunk_id = f"{book_slug}__chunk_{chunk_counter:06d}"
# #             content = it.get("content", {})
# #             # Mineru might include csv or matrix inside - best-effort stringification
# #             row = {
# #                 "id": chunk_id,
# #                 "textbook": textbook_title,
# #                 "source_file": local_filename,
# #                 "page": page,
# #                 "type": "table",
# #                 "content": json.dumps(content, ensure_ascii=False)
# #             }
# #             chunk_rows.append(row)
# #             manifest_rows.append({"id": chunk_id, "textbook": textbook_title, "page": page or "", "type": "table"})
# #             chunk_counter += 1
# #         elif it_type == "formula":
# #             chunk_id = f"{book_slug}__chunk_{chunk_counter:06d}"
# #             row = {
# #                 "id": chunk_id,
# #                 "textbook": textbook_title,
# #                 "source_file": local_filename,
# #                 "page": page,
# #                 "type": "formula",
# #                 "content": json.dumps(it.get("content", {}), ensure_ascii=False)
# #             }
# #             chunk_rows.append(row)
# #             manifest_rows.append({"id": chunk_id, "textbook": textbook_title, "page": page or "", "type": "formula"})
# #             chunk_counter += 1
# #         elif it_type == "image":
# #             # image may be present as asset; include pointer
# #             img_name = it.get("content", {}).get("file_name") or None
# #             chunk_id = f"{book_slug}__chunk_{chunk_counter:06d}"
# #             row = {
# #                 "id": chunk_id,
# #                 "textbook": textbook_title,
# #                 "source_file": local_filename,
# #                 "page": page,
# #                 "type": "image",
# #                 "content": f"IMAGE:{img_name}"
# #             }
# #             chunk_rows.append(row)
# #             manifest_rows.append({"id": chunk_id, "textbook": textbook_title, "page": page or "", "type": "image", "asset": img_name})
# #             chunk_counter += 1
# #         else:
# #             # unknown types -> store raw
# #             chunk_id = f"{book_slug}__chunk_{chunk_counter:06d}"
# #             row = {
# #                 "id": chunk_id,
# #                 "textbook": textbook_title,
# #                 "source_file": local_filename,
# #                 "page": page,
# #                 "type": it_type,
# #                 "content": str(it.get("content", ""))
# #             }
# #             chunk_rows.append(row)
# #             manifest_rows.append({"id": chunk_id, "textbook": textbook_title, "page": page or "", "type": it_type})
# #             chunk_counter += 1
# #
# #     # Save outputs
# #     jsonl_path = os.path.join(out_base, "chunks.jsonl")
# #     manifest_path = os.path.join(out_base, "manifest.csv")
# #     write_jsonl(chunk_rows, jsonl_path)
# #     save_manifest(manifest_rows, manifest_path)
# #     logger.info("Saved %d chunks to %s and manifest to %s", len(chunk_rows), jsonl_path, manifest_path)
# #     # remove file handler
# #     logger.removeHandler(fh)
# #
# # def main():
# #     token = env_token()
# #     pdfs = list_pdfs()
# #     # check TEXTBOOK_MAP: if not provided, default to filename without extension
# #     mapping = TEXTBOOK_MAP.copy()
# #     for f in pdfs:
# #         if f not in mapping:
# #             mapping[f] = pathlib.Path(f).stem
# #
# #     # We'll use ThreadPoolExecutor to process books concurrently (bounded)
# #     with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_UPLOADS) as exe:
# #         futures = []
# #         for f in pdfs:
# #             futures.append(exe.submit(process_book, f, mapping[f], token))
# #         for fut in as_completed(futures):
# #             try:
# #                 fut.result()
# #             except Exception as e:
# #                 logger.exception("Book processing failed: %s", e)
# #
# # if __name__ == "__main__":
# #     main()
#
# """
# Pipeline to upload local PDFs to Mineru, poll results,
# download parse outputs, convert to LLM-friendly chunks with textbook metadata,
# and save JSONL + manifest CSV for indexing.
# """
#
# import os
# import sys
# import time
# import json
# import csv
# import shutil
# import zipfile
# import logging
# from typing import List, Dict, Any
#
# import requests
# from tqdm import tqdm
# from dotenv import load_dotenv
#
# load_dotenv()
#
# # === CONFIG ===
# MINERU_BASE = "https://mineru.net/api/v4"
# UPLOAD_URL_BATCH = f"{MINERU_BASE}/file-urls/batch"
# BATCH_RESULT_URL = f"{MINERU_BASE}/extract-results/batch"
# SINGLE_TASK_URL = f"{MINERU_BASE}/extract/task"
# SINGLE_TASK_QUERY = f"{MINERU_BASE}/extract/task"  # GET /{task_id}
#
# PDF_DIR = "../KnowledgeBase"
# OUTPUT_DIR = "./outputs"
# MAX_CONCURRENT_UPLOADS = 3
# REQUEST_TIMEOUT = 60
# POLL_INTERVAL = 10
# MAX_POLL_WAIT = 60 * 60 * 2
# RETRY_TRIES = 5
# RETRY_BACKOFF = 2
#
# TARGET_TOKENS = 700
# TOKEN_CHAR_ESTIMATE = 4
# OVERLAP_TOKENS = 75
#
# TEXTBOOK_MAP = {
#     "applied_ml_gopal.pdf": "Applied Machine Learning by Dr. M. Gopal",
#     "intro_ml_alpaydin.pdf": "Introduction to Machine Learning by Ethem Alpaydın",
#     "machine_learning_mitchell.pdf": "Machine Learning by Tom M. Mitchell"
# }
#
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# logger = logging.getLogger("mineru_pipeline")
#
#
# def env_token() -> str:
#     tok = os.getenv("MINERU_TOKEN")
#     if not tok:
#         logger.error("Please set MINERU_TOKEN environment variable.")
#         sys.exit(1)
#     return tok
#
#
# def list_pdfs() -> List[str]:
#     files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
#     if not files:
#         logger.error("No PDFs found in %s", PDF_DIR)
#         sys.exit(1)
#     return files
#
#
# def request_with_retry(method, url, **kwargs):
#     tries = 0
#     backoff = 1
#     while tries < RETRY_TRIES:
#         try:
#             resp = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
#             resp.raise_for_status()
#             return resp
#         except Exception as e:
#             tries += 1
#             logger.warning("Request error %s %s (try %d/%d): %s", method, url, tries, RETRY_TRIES, e)
#             if tries >= RETRY_TRIES:
#                 raise
#             time.sleep(backoff)
#             backoff *= RETRY_BACKOFF
#
#
# def request_upload_urls(files: List[Dict[str, Any]], token: str) -> Dict[str, Any]:
#     url = UPLOAD_URL_BATCH
#     headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
#     payload = {
#         "enable_formula": True,
#         "language": "en",
#         "enable_table": True,
#         "files": files
#     }
#     resp = request_with_retry("POST", url, headers=headers, json=payload)
#     data = resp.json()
#     if data.get("code") != 0:
#         raise RuntimeError(f"Mineru returned error: {data}")
#     return data["data"]
#
# #
# # def upload_file_to_presigned(url: str, local_path: str):
# #     file_size = os.path.getsize(local_path)
# #     print(f"Uploading {local_path} ({file_size / (1024 * 1024):.2f} MB)...")
# #     with open(local_path, "rb") as f:
# #         with tqdm(total=file_size, unit="B", unit_scale=True, desc="Uploading", ncols=100) as pbar:
# #             uploaded = 0
# #             for chunk in iter(lambda: f.read(1024 * 1024), b""):
# #                 resp = requests.put(url, data=chunk)
# #                 uploaded += len(chunk)
# #                 pbar.update(len(chunk))
# #             pbar.close()
# #     print("✅ Upload completed successfully!")
#
# import requests
# from requests.exceptions import Timeout, ConnectionError
#
# def upload_file_to_presigned(url: str, local_path: str, max_retries: int = 3):
#     file_size = os.path.getsize(local_path)
#     print(f"Uploading {local_path} ({file_size / (1024 * 1024):.2f} MB)...")
#
#     with open(local_path, "rb") as f:
#         data = f.read()
#
#     for attempt in range(1, max_retries + 1):
#         try:
#             with tqdm(total=file_size, unit="B", unit_scale=True, desc=f"Uploading (try {attempt})", ncols=100) as pbar:
#                 chunk_size = 1024 * 1024  # 1 MB
#                 uploaded = 0
#                 for i in range(0, len(data), chunk_size):
#                     chunk = data[i:i + chunk_size]
#                     resp = requests.put(
#                         url,
#                         data=chunk,
#                         timeout=(30, 10000),  # 10s connect, 120s read
#                     )
#                     resp.raise_for_status()
#                     uploaded += len(chunk)
#                     pbar.update(len(chunk))
#             print("✅ Upload completed successfully!")
#             return
#         except (Timeout, ConnectionError) as e:
#             print(f"⚠️ Upload timeout or connection error on attempt {attempt}: {e}")
#             if attempt < max_retries:
#                 print("Retrying with a fresh presigned URL...")
#                 continue
#             else:
#                 raise
#         except requests.exceptions.HTTPError as e:
#             if e.response.status_code == 403 and attempt < max_retries:
#                 print("⚠️ Presigned URL expired, getting a new one...")
#                 continue
#             raise
#
# def safe_upload(file_obj, local_path, token: str, max_attempts: int = 3):
#     for attempt in range(max_attempts):
#         try:
#             urls_resp = request_upload_urls([file_obj], token)
#             upload_url = urls_resp["file_urls"][0]
#             upload_file_to_presigned(upload_url, local_path)
#             return urls_resp
#         except requests.exceptions.HTTPError as e:
#             if e.response.status_code == 403:
#                 logger.warning("⚠️ Presigned URL expired. Retrying attempt %d/%d...", attempt + 1, max_attempts)
#                 continue
#             raise
#     raise RuntimeError(f"Failed to upload {local_path} after {max_attempts} attempts")
#
#
# def poll_batch_result(batch_id: str, token: str, timeout_seconds: int = MAX_POLL_WAIT) -> Dict[str, Any]:
#     url = f"{BATCH_RESULT_URL}/{batch_id}"
#     headers = {"Authorization": f"Bearer {token}"}
#     start = time.time()
#     interval = POLL_INTERVAL
#     while True:
#         resp = request_with_retry("GET", url, headers=headers)
#         data = resp.json()
#         if data.get("code") != 0:
#             logger.warning("Non-zero code polling batch: %s", data)
#         extract_result = data.get("data", {}).get("extract_result")
#         if extract_result:
#             if all(r.get("state") in ("done", "failed") for r in extract_result):
#                 return data["data"]
#         if time.time() - start > timeout_seconds:
#             raise TimeoutError(f"Polling exceeded {timeout_seconds} seconds for batch {batch_id}")
#         logger.info("Batch %s not done yet. Sleeping %ds", batch_id, interval)
#         time.sleep(interval)
#         interval = min(interval * 1.5, 300)
#
#
# def download_and_extract(zip_url: str, dest_dir: str, token: str = None):
#     os.makedirs(dest_dir, exist_ok=True)
#     logger.info("Downloading %s -> %s", zip_url, dest_dir)
#     headers = {}
#     if token:
#         headers["Authorization"] = f"Bearer {token}"
#     resp = request_with_retry("GET", zip_url, headers=headers, stream=True)
#     local_zip = os.path.join(dest_dir, "mineru_result.zip")
#     with open(local_zip, "wb") as fo:
#         for chunk in resp.iter_content(chunk_size=1024 * 1024):
#             if chunk:
#                 fo.write(chunk)
#     logger.info("Downloaded zip to %s, extracting...", local_zip)
#     with zipfile.ZipFile(local_zip, "r") as zf:
#         zf.extractall(dest_dir)
#     logger.info("Extracted to %s", dest_dir)
#     return dest_dir
#
#
# def safe_slug(s: str) -> str:
#     return "".join(c if c.isalnum() else "_" for c in s)[:120]
#
#
# def estimate_tokens(text: str) -> int:
#     return max(1, int(len(text) / TOKEN_CHAR_ESTIMATE))
#
#
# def split_text_to_chunks(text: str, target_tokens: int = TARGET_TOKENS, overlap_tokens: int = OVERLAP_TOKENS):
#     paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
#     chunks = []
#     cur = ""
#     cur_tokens = 0
#     for p in paragraphs:
#         p_tokens = estimate_tokens(p)
#         if cur_tokens + p_tokens <= target_tokens or cur_tokens == 0:
#             cur += ("\n\n" if cur else "") + p
#             cur_tokens += p_tokens
#         else:
#             chunks.append(cur)
#             overlap_chars = int(overlap_tokens * TOKEN_CHAR_ESTIMATE)
#             cur = (cur[-overlap_chars:] + "\n\n" + p).strip()
#             cur_tokens = estimate_tokens(cur)
#     if cur.strip():
#         chunks.append(cur)
#     return chunks
#
#
# def load_mineru_jsons(extracted_dir: str) -> List[Dict[str, Any]]:
#     results = []
#     for root, _, files in os.walk(extracted_dir):
#         for fn in files:
#             path = os.path.join(root, fn)
#             if fn.endswith(".json"):
#                 try:
#                     with open(path, "r", encoding="utf-8") as f:
#                         data = json.load(f)
#                     results.append({"path": path, "type": "json", "data": data})
#                 except Exception as e:
#                     logger.warning("Failed to load json %s: %s", path, e)
#             elif fn.endswith((".md", ".markdown", ".html")):
#                 with open(path, "r", encoding="utf-8", errors="ignore") as f:
#                     raw = f.read()
#                 results.append({"path": path, "type": "raw", "data": raw})
#     return results
#
#
# def parse_mineru_structured_json(js: Dict[str, Any]) -> List[Dict[str, Any]]:
#     out = []
#     def walk(obj):
#         if isinstance(obj, dict):
#             t = obj.get("type")
#             if t in ("table", "Table"):
#                 out.append({"type": "table", "page": obj.get("page"), "content": obj})
#                 return
#             if t in ("formula", "Formula"):
#                 out.append({"type": "formula", "page": obj.get("page"), "content": obj})
#                 return
#             if "text" in obj and isinstance(obj["text"], str):
#                 out.append({"type": "text", "page": obj.get("page"), "content": obj["text"], "raw": obj})
#                 return
#             for v in obj.values():
#                 walk(v)
#         elif isinstance(obj, list):
#             for e in obj:
#                 walk(e)
#     walk(js)
#     return out
#
#
# def write_jsonl(rows: List[Dict[str, Any]], out_path: str):
#     with open(out_path, "w", encoding="utf-8") as fo:
#         for r in rows:
#             fo.write(json.dumps(r, ensure_ascii=False) + "\n")
#
#
# def save_manifest(manifest_rows: List[Dict[str, Any]], csv_path: str):
#     if not manifest_rows:
#         return
#     keys = list(manifest_rows[0].keys())
#     with open(csv_path, "w", newline="", encoding="utf-8") as fo:
#         writer = csv.DictWriter(fo, fieldnames=keys)
#         writer.writeheader()
#         writer.writerows(manifest_rows)
#
#
# def process_book(local_filename: str, textbook_title: str, token: str):
#     book_slug = safe_slug(textbook_title or local_filename)
#     out_base = os.path.join(OUTPUT_DIR, book_slug)
#     os.makedirs(out_base, exist_ok=True)
#
#     local_path = os.path.join(PDF_DIR, local_filename)
#     logger.info("Uploading %s -> Mineru", local_path)
#
#     file_obj = {"name": local_filename, "is_ocr": True, "data_id": textbook_title.replace(" ", "_")}
#     urls_resp = safe_upload(file_obj, local_path, token)
#     logger.info("Upload complete. Mineru will auto-submit parsing task.")
#
#     batch_id = urls_resp.get("batch_id")
#     if not batch_id:
#         upload_url = urls_resp["file_urls"][0]
#         payload = {"url": upload_url, "is_ocr": True, "enable_formula": True, "enable_table": True, "data_id": file_obj["data_id"]}
#         headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
#         resp = request_with_retry("POST", SINGLE_TASK_URL, headers=headers, json=payload)
#         d = resp.json()
#         batch_id = d["data"]["task_id"]
#         poll_url = f"{SINGLE_TASK_QUERY}/{batch_id}"
#
#         start = time.time()
#         while True:
#             r = request_with_retry("GET", poll_url, headers=headers)
#             state = r.json().get("data", {}).get("state")
#             if state == "done":
#                 full_zip = r.json()["data"]["full_zip_url"]
#                 extract_dir = os.path.join(out_base, "mineru_raw")
#                 download_and_extract(full_zip, extract_dir)
#                 break
#             elif state == "failed":
#                 raise RuntimeError("Parsing failed for task")
#             elif time.time() - start > MAX_POLL_WAIT:
#                 raise TimeoutError("Polling timeout")
#             logger.info("Single task state=%s. Sleeping...", state)
#             time.sleep(POLL_INTERVAL)
#     else:
#         batch_data = poll_batch_result(batch_id, token)
#         extract_results = batch_data.get("extract_result") or []
#         chosen = next((r for r in extract_results if r.get("file_name") == local_filename), extract_results[0])
#         full_zip = chosen.get("full_zip_url")
#         extract_dir = os.path.join(out_base, "mineru_raw")
#         download_and_extract(full_zip, extract_dir)
#
#     parsed_files = load_mineru_jsons(extract_dir)
#     items = []
#     for p in parsed_files:
#         if p["type"] == "json":
#             items.extend(parse_mineru_structured_json(p["data"]))
#         else:
#             items.append({"type": "text", "page": None, "content": p["data"], "raw_path": p["path"]})
#
#     chunk_rows, manifest_rows, chunk_counter = [], [], 0
#     for it in items:
#         it_type = it.get("type", "text")
#         page = it.get("page")
#         if it_type == "text":
#             for c in split_text_to_chunks(it.get("content", "")):
#                 chunk_id = f"{book_slug}__chunk_{chunk_counter:06d}"
#                 chunk_rows.append({"id": chunk_id, "textbook": textbook_title, "source_file": local_filename, "page": page, "type": "text", "content": c.strip()})
#                 manifest_rows.append({"id": chunk_id, "textbook": textbook_title, "page": page or "", "type": "text"})
#                 chunk_counter += 1
#
#     jsonl_path = os.path.join(out_base, "chunks.jsonl")
#     manifest_path = os.path.join(out_base, "manifest.csv")
#     write_jsonl(chunk_rows, jsonl_path)
#     save_manifest(manifest_rows, manifest_path)
#     logger.info("✅ Saved %d chunks to %s", len(chunk_rows), jsonl_path)
#
#
# def main():
#     token = env_token()
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     pdfs = list_pdfs()
#     logger.info("Found %d PDFs", len(pdfs))
#
#     for pdf in pdfs:
#         title = TEXTBOOK_MAP.get(pdf, os.path.splitext(pdf)[0])
#         logger.info("Processing %s (%s)", pdf, title)
#         try:
#             process_book(pdf, title, token)
#         except Exception as e:
#             logger.error("❌ Error processing %s: %s", pdf, e)
#     logger.info("🎉 All books processed successfully!")
#
#
# if __name__ == "__main__":
#     main()




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


# def process_book(local_filename: str, textbook_title: str, token: str):
#     book_slug = safe_slug(textbook_title)
#     out_base = os.path.join(OUTPUT_DIR, book_slug)
#     os.makedirs(out_base, exist_ok=True)
#
#     local_path = os.path.join(PDF_DIR, local_filename)
#     logger.info("Uploading %s -> Mineru", local_path)
#
#     file_obj = {"name": local_filename, "is_ocr": True, "data_id": safe_slug(textbook_title)}
#     urls_resp = safe_upload(file_obj, local_path, token)
#     logger.info("Upload complete. Mineru will auto-submit parsing task.")
#
#     batch_id = urls_resp.get("batch_id")
#     if not batch_id:
#         upload_url = urls_resp["file_urls"][0]
#         payload = {"url": upload_url, "is_ocr": True, "enable_formula": True, "enable_table": True, "data_id": file_obj["data_id"]}
#         headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
#         resp = request_with_retry("POST", SINGLE_TASK_URL, headers=headers, json=payload)
#         d = resp.json()
#         batch_id = d["data"]["task_id"]
#         poll_url = f"{SINGLE_TASK_QUERY}/{batch_id}"
#
#         start = time.time()
#         while True:
#             r = request_with_retry("GET", poll_url, headers=headers)
#             state = r.json().get("data", {}).get("state")
#             if state == "done":
#                 full_zip = r.json()["data"]["full_zip_url"]
#                 extract_dir = os.path.join(out_base, "mineru_raw")
#                 download_and_extract(full_zip, extract_dir)
#                 break
#             elif state == "failed":
#                 raise RuntimeError("Parsing failed for task")
#             elif time.time() - start > MAX_POLL_WAIT:
#                 raise TimeoutError("Polling timeout")
#             logger.info("Single task state=%s. Sleeping...", state)
#             time.sleep(POLL_INTERVAL)
#     else:
#         batch_data = poll_batch_result(batch_id, token)
#         extract_results = batch_data.get("extract_result") or []
#         chosen = next((r for r in extract_results if r.get("file_name") == local_filename), extract_results[0])
#         full_zip = chosen.get("full_zip_url")
#         extract_dir = os.path.join(out_base, "mineru_raw")
#         download_and_extract(full_zip, extract_dir)
#
#     parsed_files = load_mineru_jsons(extract_dir)
#     items = []
#     for p in parsed_files:
#         if p["type"] == "json":
#             items.extend(parse_mineru_structured_json(p["data"]))
#         else:
#             items.append({"type": "text", "page": None, "content": p["data"], "raw_path": p["path"]})
#
#     chunk_rows, manifest_rows, chunk_counter = [], [], 0
#     for it in items:
#         if it.get("type") == "text":
#             page = it.get("page")
#             for c in split_text_to_chunks(it.get("content", "")):
#                 chunk_id = f"{book_slug}__chunk_{chunk_counter:06d}"
#                 chunk_rows.append({
#                     "id": chunk_id,
#                     "textbook": textbook_title,
#                     "source_file": local_filename,
#                     "page": page,
#                     "type": "text",
#                     "content": c.strip()
#                 })
#                 manifest_rows.append({
#                     "id": chunk_id,
#                     "textbook": textbook_title,
#                     "page": page or "",
#                     "type": "text"
#                 })
#                 chunk_counter += 1
#
#     jsonl_path = os.path.join(out_base, "chunks.jsonl")
#     manifest_path = os.path.join(out_base, "manifest.csv")
#     write_jsonl(chunk_rows, jsonl_path)
#     save_manifest(manifest_rows, manifest_path)
#     logger.info("✅ Saved %d chunks to %s", len(chunk_rows), jsonl_path)
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
