from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Form, Query
import asyncpg
import os
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import shutil

import boto3
from botocore.config import Config as BotoConfig
from boto3.s3.transfer import TransferConfig

import pypdfium2 as pdfium


app = FastAPI()

API_KEY = os.environ.get("API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")

R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
# Optional override: set this if you want to paste the exact “S3 API” base
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL")  # e.g. https://<accountid>.r2.cloudflarestorage.com

# Batch sizing defaults (n8n-friendly)
DEFAULT_BATCH_PAGES = int(os.environ.get("BATCH_PAGES", "50"))
PROGRESS_EVERY_PAGES = int(os.environ.get("PROGRESS_EVERY_PAGES", "10"))

# ----------------------------
# DB helpers
# ----------------------------

_pool: asyncpg.Pool | None = None

async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    return _pool

async def db_exec(sql: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(sql, *args)

async def db_one(sql: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(sql, *args)

async def db_all(sql: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(sql, *args)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

# ----------------------------
# Schema bootstrap (safe)
# ----------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS public.pdf_extract_jobs (
  job_id uuid PRIMARY KEY,
  document_id uuid,
  changed_id text,
  file_id text,
  title text,
  source text,
  url text,

  status text NOT NULL DEFAULT 'queued', -- queued|running|done|failed
  num_pages integer,
  inserted_pages integer NOT NULL DEFAULT 0,
  error text,

  r2_key text,
  next_page integer NOT NULL DEFAULT 1,

  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  completed_at timestamptz
);

CREATE UNIQUE INDEX IF NOT EXISTS pdf_extract_jobs_doc_changed_uq
ON public.pdf_extract_jobs (document_id, changed_id);

CREATE TABLE IF NOT EXISTS public.pdf_extract_pages (
  job_id uuid NOT NULL REFERENCES public.pdf_extract_jobs(job_id) ON DELETE CASCADE,
  pdf_page integer NOT NULL,
  text text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (job_id, pdf_page)
);

CREATE INDEX IF NOT EXISTS pdf_extract_pages_job_page_idx
ON public.pdf_extract_pages (job_id, pdf_page);
"""

MIGRATE_SQL = """
ALTER TABLE public.pdf_extract_jobs
  ADD COLUMN IF NOT EXISTS r2_key text;
ALTER TABLE public.pdf_extract_jobs
  ADD COLUMN IF NOT EXISTS next_page integer NOT NULL DEFAULT 1;
"""

@app.on_event("startup")
async def _startup():
    await get_pool()
    # Create tables/indexes
    statements = [s.strip() for s in SCHEMA_SQL.split(";") if s.strip()]
    for stmt in statements:
        await db_exec(stmt)

    # Apply migrations for existing tables
    migs = [s.strip() for s in MIGRATE_SQL.split(";") if s.strip()]
    for stmt in migs:
        await db_exec(stmt)

# ----------------------------
# Auth
# ----------------------------

def require_api_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ----------------------------
# R2 client
# ----------------------------

_s3 = None

def get_r2_endpoint() -> str:
    if R2_ENDPOINT_URL:
        return R2_ENDPOINT_URL
    if not R2_ACCOUNT_ID:
        raise RuntimeError("R2_ACCOUNT_ID is not set")
    return f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

def get_s3():
    global _s3
    if _s3 is None:
        if not (R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_BUCKET_NAME):
            raise RuntimeError("R2 credentials are not set (R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/R2_BUCKET_NAME)")
        _s3 = boto3.client(
            "s3",
            endpoint_url=get_r2_endpoint(),
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto",
            config=BotoConfig(signature_version="s3v4"),
        )
    return _s3

# Multipart upload tuning (important for large PDFs)
TRANSFER_CFG = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,   # 8MB
    multipart_chunksize=8 * 1024 * 1024,   # 8MB
    max_concurrency=2,
    use_threads=True,
)

# ----------------------------
# Health
# ----------------------------

@app.get("/health")
def health():
    return {"ok": True}

# ----------------------------
# Helpers: stream upload to disk (avoid RAM), then to R2
# ----------------------------

async def save_upload_to_tmp(upload: UploadFile) -> str:
    tmp_dir = Path(tempfile.gettempdir())
    fd, tmp_path = tempfile.mkstemp(prefix="pdf_", suffix=".pdf", dir=str(tmp_dir))
    os.close(fd)

    upload.file.seek(0)
    with open(tmp_path, "wb") as out:
        shutil.copyfileobj(upload.file, out, length=1024 * 1024)  # 1MB chunks
    return tmp_path

def make_r2_key(document_id: str, changed_id: str, job_id: uuid.UUID) -> str:
    # Stable-ish path for debugging
    safe_changed = changed_id.replace("/", "_")
    return f"pdf/{document_id}/{safe_changed}/{job_id}.pdf"

def upload_file_to_r2(local_path: str, key: str):
    s3 = get_s3()
    s3.upload_file(local_path, R2_BUCKET_NAME, key, Config=TRANSFER_CFG)

def download_file_from_r2(key: str, local_path: str):
    s3 = get_s3()
    s3.download_file(R2_BUCKET_NAME, key, local_path)

# ----------------------------
# API: start job (uploads PDF to R2, does NOT process pages)
# ----------------------------

@app.post("/extract/start")
async def extract_start(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    changed_id: str = Form(...),
    file_id: str | None = Form(None),
    title: str | None = Form(None),
    source: str | None = Form("google_drive"),
    url: str | None = Form(None),
    x_api_key: str = Header(None),
):
    require_api_key(x_api_key)

    try:
        doc_uuid = uuid.UUID(document_id)
    except Exception:
        raise HTTPException(status_code=422, detail="document_id must be a UUID string")

    existing = await db_one(
        """
        SELECT job_id, status, num_pages, inserted_pages, error, r2_key, next_page
        FROM public.pdf_extract_jobs
        WHERE document_id=$1 AND changed_id=$2
        """,
        doc_uuid, changed_id
    )

    if existing and existing["status"] in ("queued", "running", "done"):
    # Merge metadata if the new request contains values (idempotent-safe)
    await db_exec(
        """
        UPDATE public.pdf_extract_jobs
        SET
          file_id = COALESCE(NULLIF($2,''), file_id),
          title   = COALESCE(NULLIF($3,''), title),
          source  = COALESCE(NULLIF($4,''), source),
          url     = COALESCE(NULLIF($5,''), url),
          updated_at = now()
        WHERE job_id = $1
        """,
        existing["job_id"],
        file_id or "",
        title or "",
        source or "",
        url or "",
    )

    # Re-read so the response reflects the updated values (optional, but nice)
    refreshed = await db_one(
        """
        SELECT job_id, status, num_pages, inserted_pages, error, file_id, title, source, url
        FROM public.pdf_extract_jobs
        WHERE job_id=$1
        """,
        existing["job_id"]
    )

    return {
        "job_id": str(refreshed["job_id"]),
        "status": refreshed["status"],
        "num_pages": refreshed["num_pages"],
        "inserted_pages": refreshed["inserted_pages"],
        "error": refreshed["error"],
        "file_id": refreshed["file_id"],
        "title": refreshed["title"],
        "source": refreshed["source"],
        "url": refreshed["url"],
        "idempotent": True,
    }

    # Create new job_id (or reuse failed job_id)
    if existing and existing["status"] == "failed":
        job_id = existing["job_id"]
        await db_exec(
            """
            UPDATE public.pdf_extract_jobs
            SET status='queued', error=NULL, updated_at=now(), started_at=NULL, completed_at=NULL
            WHERE job_id=$1
            """,
            job_id
        )
    else:
        job_id = uuid.uuid4()
        await db_exec(
            """
            INSERT INTO public.pdf_extract_jobs
              (job_id, document_id, changed_id, file_id, title, source, url, status, created_at, updated_at)
            VALUES
              ($1, $2, $3, $4, $5, $6, $7, 'queued', now(), now())
            """,
            job_id, doc_uuid, changed_id, file_id, title, source, url
        )

    # Stream upload to disk, upload to R2
    tmp_path = await save_upload_to_tmp(file)
    r2_key = make_r2_key(document_id, changed_id, job_id)

    try:
        await asyncio.to_thread(upload_file_to_r2, tmp_path, r2_key)
    except Exception as e:
        await db_exec(
            """
            UPDATE public.pdf_extract_jobs
            SET status='failed', error=$2, updated_at=now(), completed_at=now()
            WHERE job_id=$1
            """,
            job_id, f"R2 upload failed: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=f"R2 upload failed: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # Save r2_key
    await db_exec(
        """
        UPDATE public.pdf_extract_jobs
        SET r2_key=$2, updated_at=now()
        WHERE job_id=$1
        """,
        job_id, r2_key
    )

    return {"job_id": str(job_id), "status": "queued", "r2_key_saved": True}

# ----------------------------
# API: status
# ----------------------------

@app.get("/extract/status/{job_id}")
async def extract_status(job_id: str, x_api_key: str = Header(None)):
    require_api_key(x_api_key)
    try:
        job_uuid = uuid.UUID(job_id)
    except Exception:
        raise HTTPException(status_code=422, detail="job_id must be a UUID string")

    row = await db_one(
        """
        SELECT job_id, document_id, changed_id, status, num_pages, inserted_pages, next_page, error,
               created_at, updated_at, started_at, completed_at
        FROM public.pdf_extract_jobs
        WHERE job_id=$1
        """,
        job_uuid
    )
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": str(row["job_id"]),
        "document_id": str(row["document_id"]) if row["document_id"] else None,
        "changed_id": row["changed_id"],
        "status": row["status"],
        "num_pages": row["num_pages"],
        "inserted_pages": row["inserted_pages"],
        "next_page": row["next_page"],
        "error": row["error"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
    }

# ----------------------------
# API: read pages (for n8n batching downstream)
# ----------------------------

@app.get("/extract/pages/{job_id}")
async def extract_pages(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    x_api_key: str = Header(None),
):
    require_api_key(x_api_key)
    try:
        job_uuid = uuid.UUID(job_id)
    except Exception:
        raise HTTPException(status_code=422, detail="job_id must be a UUID string")

    job = await db_one(
        "SELECT status, num_pages, inserted_pages, error FROM public.pdf_extract_jobs WHERE job_id=$1",
        job_uuid
    )
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    rows = await db_all(
        """
        SELECT pdf_page, text
        FROM public.pdf_extract_pages
        WHERE job_id=$1
        ORDER BY pdf_page
        OFFSET $2 LIMIT $3
        """,
        job_uuid, offset, limit
    )

    return {
        "job_id": job_id,
        "status": job["status"],
        "num_pages": job["num_pages"],
        "inserted_pages": job["inserted_pages"],
        "error": job["error"],
        "offset": offset,
        "limit": limit,
        "pages": [{"pdf_page": r["pdf_page"], "text": r["text"]} for r in rows],
    }

# ----------------------------
# NEW: worker step endpoint (batch processing, resume-safe)
# n8n should call this in a loop until status=done
# ----------------------------

@app.post("/extract/work/{job_id}")
async def extract_work(
    job_id: str,
    max_pages: int = Query(DEFAULT_BATCH_PAGES, ge=1, le=500),
    x_api_key: str = Header(None),
):
    require_api_key(x_api_key)
    try:
        job_uuid = uuid.UUID(job_id)
    except Exception:
        raise HTTPException(status_code=422, detail="job_id must be a UUID string")

    job = await db_one(
        """
        SELECT job_id, status, num_pages, inserted_pages, error, r2_key, next_page
        FROM public.pdf_extract_jobs
        WHERE job_id=$1
        """,
        job_uuid
    )
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job["r2_key"]:
        raise HTTPException(status_code=400, detail="Job has no r2_key (did /extract/start complete?)")

    if job["status"] == "done":
        return {"job_id": job_id, "status": "done", "num_pages": job["num_pages"], "inserted_pages": job["inserted_pages"]}

    # Mark running (first work call)
    if job["status"] in ("queued", "failed"):
        await db_exec(
            """
            UPDATE public.pdf_extract_jobs
            SET status='running', started_at=COALESCE(started_at, now()), updated_at=now(), error=NULL
            WHERE job_id=$1
            """,
            job_uuid
        )

    # Download PDF from R2 to local temp file (streamed)
    tmp_dir = Path(tempfile.gettempdir())
    fd, pdf_path = tempfile.mkstemp(prefix=f"job_{job_id}_", suffix=".pdf", dir=str(tmp_dir))
    os.close(fd)

    try:
        await asyncio.to_thread(download_file_from_r2, job["r2_key"], pdf_path)

        pdf = pdfium.PdfDocument(pdf_path)
        num_pages = len(pdf)

        # Store num_pages once
        if job["num_pages"] is None:
            await db_exec(
                "UPDATE public.pdf_extract_jobs SET num_pages=$2, updated_at=now() WHERE job_id=$1",
                job_uuid, num_pages
            )

        # Determine batch start: prefer next_page, but also safe against drift:
        # if next_page points to an already-inserted page, we will skip until we find a missing one.
        start_page = int(job["next_page"] or 1)
        end_page = min(start_page + int(max_pages) - 1, num_pages)

        processed = 0
        inserted_in_this_call = 0

        for page_no in range(start_page, end_page + 1):
            # Resume-safe: only insert if missing
            exists = await db_one(
                "SELECT 1 FROM public.pdf_extract_pages WHERE job_id=$1 AND pdf_page=$2",
                job_uuid, page_no
            )
            if exists:
                processed += 1
                continue

            page = pdf.get_page(page_no - 1)
            textpage = page.get_textpage()
            text = textpage.get_text_range() or ""
            textpage.close()
            page.close()

            await db_exec(
                """
                INSERT INTO public.pdf_extract_pages (job_id, pdf_page, text)
                VALUES ($1, $2, $3)
                ON CONFLICT (job_id, pdf_page) DO NOTHING
                """,
                job_uuid, page_no, text
            )

            inserted_in_this_call += 1
            processed += 1

            if inserted_in_this_call % PROGRESS_EVERY_PAGES == 0:
                row = await db_one("SELECT COUNT(*)::int AS cnt FROM public.pdf_extract_pages WHERE job_id=$1", job_uuid)
                await db_exec(
                    "UPDATE public.pdf_extract_jobs SET inserted_pages=$2, updated_at=now() WHERE job_id=$1",
                    job_uuid, int(row["cnt"])
                )

            if inserted_in_this_call % 10 == 0:
                await asyncio.sleep(0)

        # After batch: recompute counts, advance cursor
        row = await db_one("SELECT COUNT(*)::int AS cnt FROM public.pdf_extract_pages WHERE job_id=$1", job_uuid)
        cnt = int(row["cnt"])

        new_next_page = end_page + 1
        status = "done" if cnt >= num_pages else "running"

        await db_exec(
            """
            UPDATE public.pdf_extract_jobs
            SET inserted_pages=$2,
                next_page=$3,
                status=$4,
                completed_at=CASE WHEN $4='done' THEN now() ELSE completed_at END,
                updated_at=now(),
                error=NULL
            WHERE job_id=$1
            """,
            job_uuid, cnt, new_next_page, status
        )

        return {
            "job_id": job_id,
            "status": status,
            "num_pages": num_pages,
            "inserted_pages": cnt,
            "batch_start": start_page,
            "batch_end": end_page,
            "batch_processed": processed,
            "batch_inserted": inserted_in_this_call,
            "next_page": new_next_page,
        }

    except Exception as e:
        await db_exec(
            """
            UPDATE public.pdf_extract_jobs
            SET status='failed', error=$2, updated_at=now(), completed_at=now()
            WHERE job_id=$1
            """,
            job_uuid, str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(pdf_path)
        except Exception:
            pass
